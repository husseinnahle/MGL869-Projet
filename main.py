import csv
import os
import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

from process_metrics import get_metrics, ALL_METRICS

 
DATA_DIR = Path("data")
METRICS_FILENAME = "to_release_version_%s.csv"
metrics_name = ALL_METRICS


def _get_dataset(actual_version: str, to_release_version: str, multiclass: bool) -> tuple:
    """Récupérer les données du fichier csv selon une version"""

    file = DATA_DIR / "output" / (METRICS_FILENAME % to_release_version)

    # Calculer les métriques si cela n'a pas été encore fait
    if not file.exists():
        metrics = get_metrics(to_release_version, actual_version)
        os.makedirs(file.parent, exist_ok=True)
        with open(file, "w", newline="") as file:
            writer = csv.writer(file, quotechar=None)
            writer.writerows(metrics.values.tolist())

    # Sinon lire le fichier de métriques
    else:
        metrics = pd.read_csv(file, header=None, skiprows=None, sep=",")

    # Supprimer les lignes avec des valeurs dupliquées avant de diviser les données
    metrics = metrics.drop_duplicates()

    priority = metrics.iloc[:, -1]
    metrics = metrics.iloc[:, :-1]

    # À l'origine, il existe 5 classes (0, 1, 2, 3, 4 et 5) dans le dataset.
    # Pour le problème de prédiction de bugs, fusionner les classes 1 et 2 en une seule classe.
    if not multiclass:
        mapping = {
            0: 0,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1
        }
        new_priority = priority.map(mapping)
    else:
        # Pour le problème de priorisation de fichiers à tester, fusionner les classes 1, 2, 3, 4 et 5 en 1 et 2 selon leurs présences dans le dataset.
        # Par exemple, si le dataset contient uniquement les classes 0, 1, 2 et 5, les classes 1 et 2 seront fusionnées pour finalement avoir un dataset avec les classes 0, 1 et 2.
        # Dont 0=priorité faible, 1=moyenne et 2=élevée.
        unique_priorities = sorted(list(priority.unique()))[1:]
        mapping = {
            old: 1 if old <= np.median(unique_priorities) else 2
            for old in unique_priorities
        }
        mapping[0] = 0
        new_priority = priority.map(mapping)

        # Si la fusion de classes génère une classe avec un élément, refaire la fusion avec le signe inférieur '<'.
        # Cela fera en sorte, pour l'exemple précédent, que les classes 2 et 5 seront fusionnées ensemble au lieu de 1 et 2.
        if 1 in new_priority.value_counts().values:
            mapping = {
                old: 1 if old < np.median(unique_priorities) else 2
                for old in unique_priorities
            }
            mapping[0] = 0
            new_priority = priority.map(mapping)

    return metrics, new_priority


def _pre_processing(metrics: pd.DataFrame, priority: pd.DataFrame, k_neighbors: int = 5) -> tuple:
    """Pre-traitement des données"""

    # Remplacer les valeurs manquantes par la médiane
    metrics = metrics.where(pd.notnull(metrics), None)
    imputer = SimpleImputer(strategy='median')
    metrics = imputer.fit_transform(metrics)

    # Supprimer les colonnes avec une corrélation supérieure à 0.7
    metrics_df = pd.DataFrame(metrics)
    # corr_matrix = metrics_df.corr()
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.7)]
    # metrics_df = metrics_df.drop(columns=to_drop)
    
    # Division des données en données d'entrainement et de test
    try:
        training_metrics_aug, test_metrics, training_priority_aug, test_priority = train_test_split(metrics_df, priority, test_size=0.3, random_state=42, stratify=priority)
    except ValueError:
        # Si les classes ne sont pas équilibrées, augmenter les données d'entrainement
        training_metrics_aug, test_metrics, training_priority_aug, test_priority = train_test_split(metrics_df, priority, test_size=0.5, random_state=42, stratify=priority)

    # # Augmentation des données d'entrainement
    # smote = SMOTE(k_neighbors=k_neighbors)
    # training_metrics_aug, training_priority_aug = smote.fit_resample(training_metrics, training_priority)
    return training_metrics_aug, test_metrics, training_priority_aug, test_priority


def _train_and_test(model: RandomForestClassifier | LogisticRegression, training_metrics: list, test_metrics: list, training_priority: list,
                    test_priority: list, multiclass: bool, name: str) -> tuple:
    """Entrainement et test du modèle"""

    # 10 fold cross-validation
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    training_precision = cross_val_score(model, training_metrics, training_priority, cv=kf, scoring="precision_weighted" if multiclass else "precision").mean()
    training_recall = cross_val_score(model, training_metrics, training_priority, cv=kf, scoring="recall_weighted" if multiclass else "recall").mean()

    # Entrainement
    model.fit(training_metrics, training_priority)

    # Test
    predicted = model.predict(test_metrics)
    predicted_probs = model.predict_proba(test_metrics)

    # Calcule des métriques de performance
    precision, recall, _, _ = precision_recall_fscore_support(test_priority, predicted, average="weighted" if multiclass else "binary", zero_division=0)

    # Affichage les courbes ROC par classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    classes = np.unique(test_priority)
    for i in range(len(classes)):
        binary_test_priority = np.array([1 if label == classes[i] else 0 for label in test_priority])
        fpr[i], tpr[i], _ = roc_curve(binary_test_priority, predicted_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], label=f'class {classes[i]} : AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(f'ROC curve for {name}/{model.__class__.__name__}')
    plt.show(block=False)
    question = "Q2" if multiclass else "Q1"
    plt.savefig(f'ROC_{name}_{model.__class__.__name__}_{question}.png')
    return precision, recall, training_precision, training_recall


def run(model: RandomForestClassifier | LogisticRegression, actual_version: str, to_release_version: str, multiclass: bool, k_neighbors: int = 5) -> tuple:
    """Création du modèle"""

    features, labels = _get_dataset(actual_version, to_release_version, multiclass=multiclass)
    training_features, test_features, training_labels, test_labels = _pre_processing(features, labels, k_neighbors=k_neighbors)
    precision, recall, training_precision, training_recall = _train_and_test(model, training_features, test_features, training_labels, test_labels,
                                                                             multiclass=multiclass, name=f"v.{actual_version}")
    problem = "priorisation de fichiers à tester" if multiclass else "prédiction de bugs"
    print("-----------------------------------------------------------------------------------")
    print(f"Résultats du modèle v.{actual_version}/({problem})/{model.__class__.__name__}")
    print(f"Test precision: {precision:.2f} -- Training precision: {training_precision:.2f}")
    print(f"Test recall:    {recall:.2f}    -- Training recall:    {training_recall:.2f}")
    print("-----------------------------------------------------------------------------------\n")


import warnings
warnings.filterwarnings("ignore")

# 2.0.0 -> 2.1.0
run(LogisticRegression(class_weight='balanced', max_iter=10000),    "2.0.0", "2.1.0", multiclass=True)
run(RandomForestClassifier(class_weight='balanced'),                "2.0.0", "2.1.0", multiclass=True)
run(LogisticRegression(class_weight='balanced', max_iter=10000),    "2.0.0", "2.1.0", multiclass=False)
run(RandomForestClassifier(class_weight='balanced'),                "2.0.0", "2.1.0", multiclass=False)

# 2.1.0 -> 2.2.0
run(LogisticRegression(class_weight='balanced', max_iter=10000),    "2.1.0", "2.2.0", multiclass=True)
run(RandomForestClassifier(class_weight='balanced'),                "2.1.0", "2.2.0", multiclass=True)
run(LogisticRegression(class_weight='balanced', max_iter=10000),    "2.1.0", "2.2.0", multiclass=False)
run(RandomForestClassifier(class_weight='balanced'),                "2.1.0", "2.2.0", multiclass=False)

# 2.2.0 -> 2.3.0
run(LogisticRegression(class_weight='balanced', max_iter=10000),    "2.2.0", "2.3.0", multiclass=True)
run(RandomForestClassifier(class_weight='balanced'),                "2.2.0", "2.3.0", multiclass=True)
run(LogisticRegression(class_weight='balanced', max_iter=10000),    "2.2.0", "2.3.0", multiclass=False)
run(RandomForestClassifier(class_weight='balanced'),                "2.2.0", "2.3.0", multiclass=False)

# 2.3.0 -> 3.0.0
run(LogisticRegression(class_weight='balanced', max_iter=10000),    "2.3.0", "3.0.0", multiclass=True)
run(RandomForestClassifier(class_weight='balanced'),                "2.3.0", "3.0.0", multiclass=True)
run(LogisticRegression(class_weight='balanced', max_iter=10000),    "2.3.0", "3.0.0", multiclass=False)
run(RandomForestClassifier(class_weight='balanced'),                "2.3.0", "3.0.0", multiclass=False)

# 3.0.0 -> 3.1.0
run(LogisticRegression(class_weight='balanced', max_iter=10000),    "3.0.0", "3.1.0", multiclass=True)
run(RandomForestClassifier(class_weight='balanced'),                "3.0.0", "3.1.0", multiclass=True)
run(LogisticRegression(class_weight='balanced', max_iter=10000),    "3.0.0", "3.1.0", multiclass=False)
run(RandomForestClassifier(class_weight='balanced'),                "3.0.0", "3.1.0", multiclass=False)

input("Press Enter to continue...")
