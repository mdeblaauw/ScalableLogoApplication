async function uploadImage(binary, filetype) {
    console.log('Uploading:', binary);
    
    const response = await axios({
        method: 'GET',
        url: API_URL + "/get-url",
        params: {imageType: filetype}
    });
                
    console.log('Response:', response.data);
                
    console.log('base64 string:', binary);
    let array = [];
    for (var i = 0; i < binary.length; i++) {
        array.push(binary.charCodeAt(i));
    };
                
    let blobData = new Blob([new Uint8Array(array)], {type: filetype});
                
    console.log([new Uint8Array(array)]);
                
    console.log(blobData);
    console.log(response.data.uploadURL);
                
    const result = await axios({
        method:'put',
        url:response.data.uploadURL,
        data: blobData
    });

    return response.data.imageId
}       