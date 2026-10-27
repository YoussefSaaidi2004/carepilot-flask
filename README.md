# CarePilot — application Flask d’analyse exploratoire

Interface web en français destinée à une **démonstration de recherche** autour de six tâches d’analyse à partir de `diabetic_data.csv`. Elle comprend un tableau de bord, six modules avec parcours guidé en six étapes chacun, un historique des évaluations, des rapports agrégés, un journal d’audit et une page de configuration des artefacts.

> **Important — pas un dispositif médical validé.** Les modèles sont exploratoires, entraînés sur des données historiques et non validés extérieurement. Les scores ne doivent pas décider d’un traitement, d’une sortie, d’une allocation clinique ou d’une prescription. Faites valider tout résultat par un professionnel et les données source. Le déploiement réel exige une gouvernance clinique, une validation externe, une analyse de sécurité/confidentialité et les autorisations institutionnelles applicables.

## Les six modules

1. **Durée de séjour** — XGBoost fourni dans les résultats de l’expérience XGBoost/LightGBM; cible historique `time_in_hospital >= 7`, seuil expérimental 0,27. Des variables utilisées peuvent être connues après l’admission : le score n’est donc pas une prédiction garantie « à l’admission ».
2. **Réadmission** — classifieur XGBoost optimisé pour maximiser la précision globale (~64%, ROC-AUC ~0.69), cible `readmitted != NO` (la variable regroupe ` <30` et `>30`).
3. **Charge de soins** — classe proxy apprise à partir d’un clustering K-Means sur la consommation historique de soins, puis classifiée à partir de variables de profil/admission. Ce n’est pas une vérité clinique indépendante.
4. **Profils de soins** — K-Means à trois profils de gestion; ces profils ne sont ni des types de diabète ni des recommandations thérapeutiques. La séparation des clusters est faible.
5. **Surveillance HbA1c** — classes `None`, `Norm`, `Anormal`, prédites par une architecture séquentielle en deux étages (F1 macro 0,482, ROC-AUC 0,767, Accuracy 74,9%). L'étage 1 évalue la probabilité de réalisation du test et l'étage 2 évalue le risque d'anomalie glycémique conditionnel.
6. **Profil thérapeutique** — classifie un motif historique associé à la présence d’insuline et/ou de metformine. **Ce module ne recommande pas de traitement.**

Le modèle de surveillance HbA1c intègre une architecture séquentielle à deux étages avec décomposition du dépistage et du risque glycémique, explicabilité TreeSHAP et protocole de conduite à tenir ADA.

## Prérequis

- Python 3.11–3.14 (validation locale effectuée avec Python 3.14)
- Les dépendances de `requirements.txt`
- Le dossier `models/` inclus avec ce projet

## Installation sous Windows (PowerShell)

```powershell
cd carepilot_flask
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Ouvrez `.env`, puis collez une clé aléatoire générée avec `python -c "import secrets; print(secrets.token_urlsafe(48))"` dans `SECRET_KEY`. L’application fonctionne en mode local et ouvre directement le dashboard ; aucun écran de connexion n’est requis :

```powershell
python app.py
```

Ouvrez `http://127.0.0.1:5000`. Le premier lancement crée `instance/carepilot.sqlite3`.

## Installation Linux / macOS

```bash
cd carepilot_flask
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# Éditer .env et remplacer SECRET_KEY
flask --app app create-admin
python app.py
```

## Dossiers

- `app.py` — routes Flask, authentification, CSRF, historique, audit, CLI.
- `domain.py` — description des tâches, champs et validation métier.
- `ml_service.py` — conversion des entrées vers les modèles et calculs.
- `train_models.py` — reconstruction des modèles auxiliaires sur la base du CSV et sauvegarde des options.
- `models/` — artefacts de modèles (`joblib`) et manifeste de provenance/limites.
- `data/feature_options.json` — valeurs catégorielles autorisées par les formulaires et par l’inférence.
- `templates/`, `static/` — UI responsive, animations et interactions. Les styles et icônes sont locaux; aucun appel vers une police ou un CDN externe n’est requis.

## Reconstruire les modèles auxiliaires

Le modèle XGBoost comparatif fourni doit être présent à `models/modeles_comparaison.joblib`. Avec le CSV original disponible localement :

```bash
python train_models.py --csv /path/to/diabetic_data.csv --out models
```

Cette commande entraîne les modèles auxiliaires et met à jour `data/feature_options.json`. Les artefacts `joblib` ne doivent être chargés que si vous faites confiance à leur provenance.

## Contrôle de bon fonctionnement

Après l’installation des dépendances, lancez `python smoke_test.py`. Ce test utilise uniquement des valeurs synthétiques et vérifie que les six artefacts répondent. Il ne teste pas la validité clinique.

## Contrôles déjà prévus dans l’interface

- Entrées catégorielles sélectionnées dans des listes contrôlées, choix segmentés, cartes de sélection, boutons numériques bornés et recherche assistée pour les codes de diagnostic.
- Validation côté navigateur **et** validation côté serveur : plage d’entiers, liste blanche des valeurs catégorielles et validation de la référence pseudonymisée.
- CSRF sur les formulaires, mots de passe hachés, cookies HTTP-only / SameSite, sessions à durée limitée.
- Pas de nom ou de numéro national requis. Les valeurs brutes du formulaire sont stockées temporairement côté serveur dans SQLite pour permettre le parcours multi-étapes, avec expiration après une heure d’inactivité et suppression au calcul réussi ou à la déconnexion. Les choix de traitement et les résultats de laboratoire commencent sans sélection automatique; le formulaire distingue « Non » de « Non documenté ». Pour les compteurs, « 0 » reste distinct de « Valeur non documentée »; pour le clustering, les valeurs inconnues sont imputées selon les médianes du jeu d’entraînement et ne sont pas interprétées comme un signal positif. L’historique d’évaluations ne conserve que le résultat synthétique et la référence pseudonymisée optionnelle. SQLite n’est pas chiffrée par défaut : n’utilisez pas de données personnelles ou réelles dans cette démonstration.

## Préparation d’un vrai déploiement

Ne rendez pas cette version accessible à des utilisateurs cliniques ni sur Internet sans au minimum : HTTPS/TLS, clé secrète forte, `COOKIE_SECURE=1`, serveur WSGI de production, gestion des rôles, rotation des journaux et sauvegardes contrôlées, tests de sécurité, analyse de risques, minimisation/rétention des données, validation externe par site, surveillance des dérives et gouvernance institutionnelle. N’utilisez pas le serveur de développement Flask en production.
