const API_URL = 'https://bvf8r66553.execute-api.eu-west-1.amazonaws.com/prod/get-url'

function getBinary(input) {
    const file = input.files[0];
    const reader = new FileReader();
    reader.readAsBinaryString(file);
    reader.onload = function () {
        return uploadImage(reader.result, file.type);
    }
    reader.onerror = function (error) {
        console.log('Error: ', error);
    }
}
            
async function uploadImage(binary, filetype) {
    console.log('Uploading:', binary);
    
    const response = await axios({
        method: 'GET',
        url: API_URL,
        params: {imageType: filetype}
    })
                
    console.log('Response:', response.data);
                
    console.log('base64 string:', binary);
    let array = [];
    for (var i = 0; i < binary.length; i++) {
        array.push(binary.charCodeAt(i));
    }
                
    let blobData = new Blob([new Uint8Array(array)], {type: filetype});
                
    console.log([new Uint8Array(array)]);
                
    console.log(blobData);
    console.log(response.data.uploadURL);
                
    const result = await axios({
        method:'put',
        url:response.data.uploadURL,
        data: blobData
    })
}       