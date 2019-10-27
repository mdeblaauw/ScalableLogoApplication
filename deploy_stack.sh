#!/bin/bash

input1="$1"
input2="$2"
bucketName=logoapplication-lambdafiles-34fl9as0
stackName=logoApplication-${input1}
templateFile=infrastructure.yaml
outTemplateFile=infrastructure.packaged.yaml

aws cloudformation package --template-file $templateFile --s3-bucket $bucketName --output-template-file $outTemplateFile
aws cloudformation deploy --template-file $outTemplateFile --stack-name $stackName --parameter-overrides EnvType=$input1 ProdVersion=$input2 --capabilities "CAPABILITY_IAM"

echo This is the get image url API
aws cloudformation list-exports --query "Exports[?Name==\`GetApiURL-${input1}\`].Value"

echo This is the website url
aws cloudformation list-exports --query "Exports[?Name==\`WebsiteURL-${input1}\`].Value"

echo This is the name of the preparation bucket
aws cloudformation list-exports --query "Exports[?Name==\`PrepBucketName-${input1}\`].Value"




