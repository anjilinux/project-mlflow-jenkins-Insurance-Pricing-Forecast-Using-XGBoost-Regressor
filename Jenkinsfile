pipeline {
    agent any

    stages {

        stage('Setup Environment') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }

        stage('Data Preprocessing') {
            steps {
                sh '''
                . venv/bin/activate
                python preprocess.py
                '''
            }
        }
       

        stage('Clean MLflow') {
            steps {
                sh '''
                rm -rf /var/lib/jenkins/mlflow
                mkdir -p /var/lib/jenkins/mlflow
                '''
            }
        }



        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python train.py
                '''
            }
        }

        stage('Evaluate Model') {
            steps {
                sh '''
                . venv/bin/activate
                python evaluate.py
                '''
            }
        }

        stage('Test Model') {
            steps {
                sh '''
                . venv/bin/activate
                pytest test_model.py
                '''
            }
        }
    }
}
