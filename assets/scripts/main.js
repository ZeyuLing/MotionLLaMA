let idx = 0;

function cycleGallery() {
    let rvid = document.getElementById("rotatingvideo" + (idx + 1));
    let rcap = document.getElementById("rotatingcaption" + (idx + 1));
    rvid.classList.add("hidden");
    rcap.classList.add("hidden");
    idx = (idx + 1) % (4);
    rvid = document.getElementById("rotatingvideo" + (idx + 1));
    rvid.classList.remove("hidden");
    rcap = document.getElementById("rotatingcaption" + (idx + 1));
    rcap.classList.remove("hidden");
}