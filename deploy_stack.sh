#!/bin/bash

input1="$1"
bucketName=logoapplication-lambdafiles-34fl9as0
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

    echo This is the get image url API
    aws cloudformation list-exports --query "Exports[?Name==\`GetApiURL\`].Value"

    echo This is the website url
    aws cloudformation list-exports --query "Exports[?Name==\`WebsiteURL\`].Value"

    echo This is the name of the preparation bucket
    aws cloudformation list-exports --query "Exports[?Name==\`PrepBucketName\`].Value"
fi



