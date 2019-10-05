async function add_logo_images() {
    const response = await axios({
        method: 'GET',
        url: API_URL + "/get-support-set"
    });

    var files = new Array();
    var labels = new Array();

    const body = response.data;

    for( let prop in body ){
        console.log( body[prop] );
        files.push(body[prop]);
        labels.push(body[prop].split('/').slice(-1)[0].split('.').slice(-2)[0]);
    };

    for (i=0; i<files.length; i++) {
        const elem = document.createElement('figure');
        const label = document.createElement('figcaption');
        const elemText = document.createTextNode(labels[i]);
        label.appendChild(elemText);
        const img = document.createElement('img');
        img.setAttribute('src', files[i]);
        img.setAttribute('alt', labels[i]);
        elem.appendChild(label);
        elem.appendChild(img);
        document.getElementById('classes').appendChild(elem);
    }
};