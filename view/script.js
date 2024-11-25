function sendText() {
    var text = document.getElementById("inputText").value;
    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("intent").innerText = "Intent: " + data.intent;
        document.getElementById("entities").innerText = "Entities: " + JSON.stringify(data.entities);
    });
}
