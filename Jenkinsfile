pipeline {
agent any


stages {
stage('Setup Venv') {
steps {
sh 'python3 -m venv venv'
sh 'venv/bin/pip install -r requirements.txt'
}
}


stage('Preprocess Data') {
steps {
sh 'venv/bin/python src/preprocess.py'
}
}


stage('Train Model') {
steps {
sh 'venv/bin/python src/train.py'
}
}


stage('Test Model') {
steps {
sh 'venv/bin/pytest tests/'
}
}
}
}