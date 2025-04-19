document.getElementById("analyze-btn").addEventListener("click", async () => {
  const text = document.getElementById("text-input").value;
  const fileInput = document.getElementById("image-input");
  const resultDiv = document.getElementById("result");

  const formData = new FormData();
  formData.append("text", text);

  if (fileInput.files.length > 0) {
      formData.append("image", fileInput.files[0]);
  }

  try {
      const response = await fetch("http://127.0.0.1:5000/analyze", {
          method: "POST",
          body: formData
      });

      const data = await response.json();

      resultDiv.innerHTML = `<strong>Sentiment:</strong> ${data.sentiment} <br><strong>Extracted Text:</strong> ${data.text}`;
  } catch (error) {
      console.error("Error:", error);
      resultDiv.innerHTML = "❌ Something went wrong. Check the server or console.";
  }
});
