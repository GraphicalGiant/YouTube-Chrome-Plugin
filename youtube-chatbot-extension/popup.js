

document.getElementById("ask").addEventListener("click", () => {
  const question = document.getElementById("question").value;

  // Get the current YouTube video tab
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    const url = new URL(tabs[0].url);
    const videoId = url.searchParams.get("v");

    fetch("https://mychatbot-api.onrender.com/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: videoId,
        question: question
      })
    })
    .then(res => res.json())
    .then(data => {
      document.getElementById("answer").innerText = data.answer;
    })
    .catch(err => {
      document.getElementById("answer").innerText = "Error: " + err.message;
    });
  });
});
