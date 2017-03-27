document.querySelectorAll("img").forEach((img) => {
   if (img.currentSrc)
      img.src = "http://localhost:8000/image?path=" + encodeURIComponent(img.currentSrc);
});
