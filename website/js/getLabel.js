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
                console.log(response.data);
                const probLabel = response.data.ProbLabel.S;
                let maxProb = response.data.MaxProb.N * 100;
                const minDist = response.data.MinDist.N;
                spinner_switch("hide");
                const answer = `The model is ${maxProb.toFixed(0)}% sure that the ${probLabel} logo is in the submitted picture. (Min distance is ${minDist})`;
                document.getElementById("output").innerHTML = answer;
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