const delayMs = 2000;

async function getLabel(imageId) {
    console.log(imageId);

    const delay = function(ms) {
        return new Promise(function(resolve) {
            setTimeout(resolve, ms);
         })
    };

     async function getImageLabel() {
        await delay(delayMs);
        let count = 0;
        while (count < 7){
            const response = await axios({
                method: 'GET',
                url: API_URL + "/get-label",
                params: {imageId: imageId}
            });
            if(response.data) {
                document.getElementById("label").innerHTML = response.data.ProbLabel.S
                document.getElementById("probability").innerHTML = response.data.MaxProb.N
                return console.log(response.data.ProbLabel.S);
            }
            console.log(count);
            count++
            await delay(delayMs);
        }
    }

    getImageLabel();

    //console.log(answer);
    //console.log(answer.data.ProbLabel.S);

}