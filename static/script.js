const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  // Add user bubble
  addMessage("You", message, "user");
  userInput.value = "";

  // Add loading indicator
  const loadingId = addMessage("Bot", "⏳ Thinking...", "bot");

  try {
    const response = await fetch("http://127.0.0.1:8000/chat/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    const data = await response.json();

    // Remove loading
    document.getElementById(loadingId).remove();

    // Add bot reply
    addMessage("Bot", `${data.response}`, "bot", data.route);

  } catch (error) {
    document.getElementById(loadingId).remove();
    addMessage("Bot", "❌ Failed to fetch response", "bot");
  }
}

function addMessage(sender, text, type, route = null) {
  const msgId = "msg-" + Date.now();
  const div = document.createElement("div");
  div.className = `message ${type}`;
  div.id = msgId;

  div.innerHTML = `
    <div class="bubble">
      <strong>${sender}:</strong> ${text}
      ${route ? `<span class="route">(${route})</span>` : ""}
    </div>
  `;

  chatBox.appendChild(div);
  chatBox.scrollTo({
  top: chatBox.scrollHeight,
  behavior: "smooth"
});
  return msgId;
}

// Event listeners
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});
clearBtn.addEventListener("click", () => {
  chatBox.innerHTML = "";
});
