window.onload = function() {
    console.log("asdafdsfasd");
    var heatmap = h337.create({
  container: document.body
});

heatmap.setData({
  max: 5,
  data: [{ x: 10, y: 15, value: 5}]
});
}
document.querySelectorAll("img").forEach((img) => {
    if (img.currentSrc)
        img.src = "http://localhost:8000/image?path=" + encodeURIComponent(img.currentSrc);
});
