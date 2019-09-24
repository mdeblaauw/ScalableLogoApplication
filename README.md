# ScalableLogoApplication

This serverless application is an implementation of my thesis work. Wherein was researched if a logo classifier can be prepared on any logo class with only five logo samples.
To run this application a few prerequisites are needed:

1. Install AWS CLI and configure your AWS account.
2. Download this repo.
3. Download a lambda python dependency package from this [link](https://drive.google.com/open?id=1t4VVur5mjhyfvp5k9yv98PKNBhg-Utfg) and put it in folder lambda_functions/model_function/.
4. Create an S3 bucket with name 'logoapplication-lambdafiles-34fl9as0'. This bucket stores the lambda functions.
5. Run deploy_stack.sh. This implements the serverless application with cloudformation (https://aws.nz/best-practice/cloudformation-package-deploy/).
6. Put the output URL of the upload image api in the 'website/uploadImage.js' file.
7. Run deploy_website.sh, which uploads the website contents to an S3 bucket.
