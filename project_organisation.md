├── JUIN23_CONTINU_MLOPS_POMPIERS    
│
├── .github\workflows
│     └── python-app.yml
├── airflow
│   ├── dags
│   │   └── dag_mlops_pompiers.py
│   ├── docker-compose.yaml
│   └── models
│       └── null_file.json
├── docker-compose.yml
├── LICENSE
├── models
├── notebooks
│   ├── 1.0-wm-data-exploration-and-testing-models.ipynb
│   ├── 2.0-wm-data-exploration-and-testing-models.ipynb
│   ├── 2.1-wm-testing-models-features-selection.ipynb
│   └── 2.2-wm-random-forest-features-selection.ipynb
├── README.md
├── references
│   ├── Cahier des charges LFB.docx
│   ├── Diagram-light.png
│   ├── Diagram-MLOps-pompiers
│   ├── Diagram-MLOps-pompiers.drawio.png
│   ├── Metadata
│   │   ├── Incidents Metadata.xlsx
│   │   └── Mobilisations Metadata.xlsx
│   └── Prédiction du temps de réponse des pompiers.docx
├── requirements.txt
└── src
    ├── api_admin
    │   ├── api
    │   │   ├── __init__.py
    │   │   ├── schema.py
    │   │   └── users.py
    │   ├── api_admin.py
    │   ├── data
    │   │   ├── import_raw_data.py
    │   │   ├── __init__.py
    │   │   ├── make_dataset.py
    │   ├── Dockerfile
    │   ├── models_training
    │   │   ├── __init__.py
    │   │   ├── model.py
    │   ├── __pycache__
    │   │   ├── api_admin.cpython-310.pyc
    │   │   └── test_api_admin.cpython-310-pytest-7.4.0.pyc
    │   ├── requirements.txt
    │   ├── test_api_admin.py
    │   └── tests
    │       ├── __init__.py
    │       ├── test_import_raw_data.py
    │       └── test_model.py
    └── api_user
        ├── api
        │   ├── fonction.py
        │   ├── __init__.py
        │   ├── schema.py
        │   └── users.py
        ├── api_user.py
        ├── data
        │   ├── import_raw_data.py
        │   ├── __init__.py
        │   └── make_dataset.py
        ├── Dockerfile
        ├── models_training
        │   ├── __init__.py
        │   └── model.py
        ├── requirements.txt
        └── test_api_user.py
