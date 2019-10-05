const uuidv4 = require('uuid/v4'); //makes unique image identifier
const AWS = require('aws-sdk');
AWS.config.update({
  region: process.env.REGION, 
  signatureVersion: 'v4',
  });
const s3 = new AWS.S3();
const ddb = new AWS.DynamoDB({apiVersion: '2012-08-10'});

exports.handler = async (event, context) => {  
  if (event.path === '/get-url') {
    const imageType = event.queryStringParameters.imageType;
    return await getUploadURL(imageType);
  } else if (event.path === '/get-support-set') {
    return await getSupportURLs();
  }else {
    const imageId = event.queryStringParameters.imageId;
    return await getLabel(imageId);
  }
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
          "imageId": `${actionId}`
      })
    });
  });
};

const getLabel = async function(imageId) {
  console.log(imageId)
  var params = {
    TableName: process.env.TableName,
    Key: {
      'ImageId': {S: imageId}
    },
    ProjectionExpression: 'MaxProb, ProbLabel'
  };

  // Call DynamoDB to read the item from the table and return
  return new Promise((resolve, reject) => {
    ddb.getItem(params, function(err, data) {
      if (err) {
        return reject(err);
      } else {
        return resolve({
          "statusCode": 200,
          "isBase64Encoded": false,
          "headers": {
            "Access-Control-Allow-Origin": "*"
          },
            "body": JSON.stringify(data.Item)
          });
      }
      
    });
  })
};

const getSupportURLs = async function () {
  const params = {
    Bucket: process.env.websiteBucket,
    Prefix: 'preparation-images/'
  };
  
  const data = await s3.listObjectsV2(params).promise();
  const object = {};
  for (var i = 0; i < data.Contents.length; i++) {
    object[`file${i}`] = data.Contents[i].Key;
  }

  return new Promise((resolve, reject) => {
    return resolve({
      "statusCode": 200,
      "isBase64Encoded": false,
      "headers": {
        "Access-Control-Allow-Origin": "*"
      },
      "body": JSON.stringify(object)
    });
  });
};

