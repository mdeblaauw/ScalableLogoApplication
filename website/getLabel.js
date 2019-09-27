async function getLabel(imageId) {
    console.log(imageId);

    const response = await axios({
        method: 'GET',
        url: API_URL + "/get-label",
        //params: {imageId: this.imageId}
        params: {imageId: "8cb23bb5-e8b6-4e4e-a226-58c186488711"}
    });

    console.log(response);
    console.log(response.data.ProbLabel.S);

}