#!/usr/bin/groovy

@Library(['github.com/indigo-dc/jenkins-pipeline-library@1.4.0']) _

def job_result_url = ''

ci_cd_image = 'mteamkit/cicd-python-gl'
ci_cd_image_registry = 'https://docker.io'

pipeline {
    agent {
        docker { 
            image "${ci_cd_image}"
            registryUrl "${ci_cd_image_registry}"
            registryCredentialsId 'indigobot'
        }
    }

    environment {
        author_name = "Fahimeh"
        author_email = "khadijeh.alibabaei@kit.edu"
        app_name = "yolov8_api"
        job_location = "Pipeline-as-code/DEEP-OC-org/DEEP-OC-yolov8_api/${env.BRANCH_NAME}"
    }

    stages {
        stage('Code fetching') {
            steps {
                checkout scm
            }
        }

        stage('Metrics gathering') {
            steps {
                //SLOCRun()
                sh "cloc --by-file --fullpath --not-match-d='(.tox|htmlcov|.egg-info)' --xml --out=cloc.xml ."
            }
            post {
                success {
                    SLOCPublish()
                }
            }
        }

        stage('Style analysis: PEP8') {
            steps {
                ToxEnvRun('qc.sty')
            }
            post {
                always {
                    recordIssues(tools: [flake8(pattern: 'flake8.log')])
                }
            }
        }

        stage('Coverage analysis: pytest-cov') {
            steps {
                ToxEnvRun('qc.cov')
            }
            post {
                success {
                    HTMLReport('htmlcov', 'index.html', 'coverage.py report')
                }
            }
        }

        stage('Security scanner: bandit') {
            steps {
                ToxEnvRun('qc.sec')
                script {
                    if (currentBuild.result == 'FAILURE') {
                        currentBuild.result = 'UNSTABLE'
                    }
               }
            }
            post {
               always {
                    HTMLReport("./", 'bandit.html', 'Bandit report')
                }
            }
        }

        stage("Re-build Docker images") {
            when {
                anyOf {
                   branch 'main'
                   branch 'test'
                   buildingTag()
               }
            }
            steps {
                script {
                    def job_result = JenkinsBuildJob("${env.job_location}")
                    job_result_url = job_result.absoluteUrl
                }
            }
        }

    }



}
