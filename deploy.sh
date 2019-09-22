#!/bin/bash

aws cloudformation package --template-file infrastructure.yaml --s3-bucket blaauwtestbucket --output-template-file infrastructure.packaged.yaml
aws cloudformation deploy --template-file infrastructure.packaged.yaml --stack-name logoApplication --capabilities "CAPABILITY_IAM"

