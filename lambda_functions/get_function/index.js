const uuidv4 = require('uuid/v4'); //makes unique image identifier
const AWS = require('aws-sdk');
AWS.config.update({
  region: process.env.REGION, 
  signatureVersion: 'v4',
  });
const s3 = new AWS.S3();

exports.handler = async (event, context) => {
  const imageType = event.queryStringParameters.imageType;
  
  const result = await getUploadURL(imageType);
  return result;
};

const getUploadURL = async function(imageType) {
  let actionId = uuidv4();
  let extension = 'jpg';
  
  if (imageType === 'image/png') {
      extension = 'png';
    }
  
  var s3Params = {
    Bucket: process.env.uploadBucket,
    ContentType: imageType,
    Key:  `${actionId}.${extension}`,
    Expires: 60,                      //url expires after 1 minute
    ACL: 'public-read',
  };

  return new Promise((resolve, reject) => {
    // Get signed URL
    let uploadURL = s3.getSignedUrl('putObject', s3Params);
    
    resolve({
      "statusCode": 200,
      "isBase64Encoded": false,
      "headers": {
        "Access-Control-Allow-Origin": "*"
      },
      "body": JSON.stringify({
          "uploadURL": uploadURL,
          "photoFilename": `${actionId}.${extension}`
      })
    });
  });
};

