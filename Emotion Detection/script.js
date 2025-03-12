async function predictEmotion() {
    const text = document.getElementById("textInput").value;
    if (!text) {
        alert("Please enter text.");
        return;
    }

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text }),
    });

    const data = await response.json();
    let emotions = data.emotions;
    let resultText = "<h3>Emotion Scores:</h3>";
    for (let [emotion, score] of Object.entries(emotions)) {
        resultText += `<p><b>${emotion}:</b> ${score.toFixed(2)}</p>`;
    }

    document.getElementById("result").innerHTML = resultText;
}
