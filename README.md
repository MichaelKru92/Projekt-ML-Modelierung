# Thema 6. ML-Modellierung für die Prädiktion klinischer Ereignisse mit besonderem Fokus auf Versionskontrolle und Validierung
## Forschungshintergrund
Machine Learning in der Medizin erfordert nicht nur ein gutes Modell, sondern auch saubere Daten, Nachvollziehbarkeit (Data Provenance), eine sorgfältige Validierung und Versionierung – sowohl der Daten als auch der Modelle. Ein gutes Datenmanagement ist hier entscheidend, um reproduzierbare Forschung zu gewährleisten.
## Mögliche Zielsetzungen
**1. Datenvorverarbeitung:** Aufbau einer strukturierten Pipeline (z. B. in Python), die Rohdaten reinigt, anreichert und Features extrahiert.
**2. Versionierung (Data & Model):** Einführung von Tools wie DVC (Data Version Control) oder Git LFS, um Daten- und Modellversionen in verschiedenen Stadien zu tracken.
**3. Modellvalidierung:** Umsetzung eines strengen Validierungs- und Testprozesses (z. B. Cross-Validation, ROC-Kurven, Metrikenreport).
**4. CICD & Deployment:** Entwurf eines vereinfachten Continuous-Integration-Konzepts (z. B. GitHub Actions) für automatisierte Trainingsläufe und optionales Deployment in einer Testumgebung.
### Mögliche Datenquellen
- MIMIC-IV (aktuellere Version der ICU-Datenbank mit umfangreichen klinischen Parametern)
- Kaggle Medical ML Datasets
- UCI ML Repository – Medical Prognosis Datasets
