let model;

const labels = ["Apple", "Banana", "Orange"];

async function loadModel() {
    model = await tf.loadLayersModel('model/model.json');
    console.log("Model Loaded");
}

loadModel();

const imageUpload = document.getElementById("imageUpload");
const previewImage = document.getElementById("previewImage");

imageUpload.addEventListener("change", function(event) {
    const file = event.target.files[0];

    if (file) {
        previewImage.src = URL.createObjectURL(file);
    }
});

document.getElementById("predictBtn")
.addEventListener("click", async () => {

    if (!previewImage.src) {
        alert("Please upload an image.");
        return;
    }

    const tensor = tf.browser.fromPixels(previewImage)
        .resizeNearestNeighbor([128, 128])
        .toFloat()
        .div(255.0)
        .expandDims();

    const prediction = model.predict(tensor);

    const data = await prediction.data();

    let maxIndex = data.indexOf(Math.max(...data));

    document.getElementById("result").innerHTML =
        `Prediction: ${labels[maxIndex]}`;
});
