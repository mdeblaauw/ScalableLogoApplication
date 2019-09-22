#!/bin/bash

input1="$1"
bucketName=blaauwtestbucket
stackName=logoApplication
templateFile=infrastructure.yaml
outTemplateFile=infrastructure.packaged.yaml

if [ $input1 == "update" ]; then
    aws cloudformation update-stack --stack-name $stackName
elif [ $input1 == "delete" ]; then
    echo deleting stack
    aws cloudformation delete-stack --stack-name $stackName
    echo deleting succeeded
else
    aws cloudformation package --template-file $templateFile --s3-bucket $bucketName --output-template-file $outTemplateFile
    aws cloudformation deploy --template-file $outTemplateFile --stack-name $stackName --capabilities "CAPABILITY_IAM"
fi



