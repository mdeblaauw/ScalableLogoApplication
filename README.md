# ScalableLogoApplication

This serverless application is an implementation of my thesis work. Wherein is researched if a logo classifier can be prepared on any logo class with only five logo samples.
To run this application a few prerequisites are needed:

1. Install AWS CLI and configure your AWS account.
2. Download this repo.
3. Create an S3 bucket with name 'logoApplicationFiles-34fl9as0'. This bucket stores the lambda functions.
4. Run deploy_stack.sh. This implements the serverless application with cloudformation (https://aws.nz/best-practice/cloudformation-package-deploy/).
5. Put the output URL of the upload image api in the 'website/uploadImage.js' file.
6. Run deploy_website.sh, which uploads the website contents to an S3 bucket.
