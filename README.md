# ScalableLogoApplication

This serverless application is an implementation of my thesis work. Wherein was researched if a logo classifier can be prepared on any logo class with only five logo samples.
To run this application a few prerequisites are needed:

1. Install AWS CLI and configure your AWS account.
2. Download this repo.
3. Download a lambda python dependency package from this [link](https://drive.google.com/open?id=1t4VVur5mjhyfvp5k9yv98PKNBhg-Utfg) and put the unzipped folders in directory lambda_functions/model_function/lambda_dependencies/pytorch1_0_1/.
4. Create an S3 bucket with name 'logoapplication-lambdafiles-34fl9as0'. This bucket stores the lambda functions.
5. Run `bash deploy_stack.sh <EnvType> <ProdVersion>`, with `<EnvType>` either dev or prod and `<ProdVersion>` the production version, such as v1.0. This implements the serverless application with cloudformation (https://aws.nz/best-practice/cloudformation-package-deploy/).
6. Download the model weigths from [link](https://drive.google.com/file/d/1T8aWML4vbwUROehLtVklqUCu13b7THLn/view?usp=sharingPut) and put them in the preparation bucket by using prefix 'preparation bucket name'/model-weights/.
7. Zip the support set with command 'zip -r preparation.zip support_set/' and dump it in the preparation bucket.
8. Put the output API URL in the 'website/index.html' file.
9. Dump the content of the website folder in the website bucket.

Note1: the prod deployment uses the domain name `scalablelogorecognition.com` and is purchased for one year as of 25-10-2019. It is set accordingly in Route53 and hence only the appropiate buckets need to be created to let the application function correctly.

Note2: prod application invokes a Bucket and DynamoDB that have a retain deletion policy.
