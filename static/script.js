let chatHistory = [];

async function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    if (!userInput.trim()) return;

    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<div><strong>You:</strong> ${userInput}</div>`;
    document.getElementById("user-input").value = "";

    try {
        const response = await fetch("http://127.0.0.1:8000/chat/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userInput, history: chatHistory })
        });
        const data = await response.json();
        // The backend returns { response: ..., history: [...] }
        chatBox.innerHTML += `<div><strong>Bot:</strong> ${data.response}</div>`;
        // Update chatHistory with the new history from backend (if present)
        if (data.history) {
            chatHistory = data.history;
        } else {
            // Fallback: push the latest turn
            chatHistory.push([userInput, data.response]);
        }
    } catch (error) {
        chatBox.innerHTML += `<div><strong>Error:</strong> Failed to fetch response</div>`;
    }
    chatBox.scrollTop = chatBox.scrollHeight;
}
