# Seattle - Projet Machine Learning (D-CLIC)

Ce dossier contient la version partageable du projet Seattle pour D-CLIC:
- donnees,
- notebooks,
- modeles entraines,
- prototype d'inference.

## Contenu

- `2016_Building_Energy_Benchmarking.csv`
- `[D-CLIC phase3] Projet Machine Learning - Apprentissage supervisé - Niveau intermédiaire.pdf`
- `deliverables/01_eda.ipynb`
- `deliverables/02_model_co2.ipynb`
- `deliverables/03_model_energy.ipynb`
- `deliverables/models/co2_final_model.joblib`
- `deliverables/models/energy_final_model.joblib`
- `deliverables/script_oral_presentation.pdf`
- `prototype/app.py`
- `prototype/requirements.txt`
- `prototype/sample_input.csv`

## Ce qui n'est pas inclus

Les fichiers de presentation (`.pptx`) ne sont pas inclus dans cette version.

## Lancer le prototype

Depuis la racine du dossier `Seattle`:

```bash
pip install -r prototype/requirements.txt
streamlit run prototype/app.py
```

## Utilisation rapide

1. Ouvrir l'onglet **Prediction unitaire** pour une estimation ponctuelle.
2. Ouvrir **Comparaison scenarios** pour tester l'impact d'un changement (A vs B).
3. Ouvrir **Prediction batch CSV** pour predire en masse puis exporter les resultats.

## Notes

- Les modeles utilisent 6 features principales en entree UI.
- L'application calcule et affiche aussi l'intensite energetique predite (`kBtu/sf`) et son percentile sur la distribution de reference.
