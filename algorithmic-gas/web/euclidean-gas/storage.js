async function database() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("fragile-algorithmic-gas", 1);
    request.onupgradeneeded = () =>
      request.result.createObjectStore("checkpoints");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
export async function saveCheckpoint(bytes) {
  const db = await database();
  try {
    await new Promise((resolve, reject) => {
      const tx = db.transaction("checkpoints", "readwrite");
      tx.objectStore("checkpoints").put(bytes, "latest");
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}
export async function loadCheckpoint() {
  const db = await database();
  try {
    return await new Promise((resolve, reject) => {
      const request = db
        .transaction("checkpoints")
        .objectStore("checkpoints")
        .get("latest");
      request.onsuccess = () =>
        request.result
          ? resolve(request.result)
          : reject(new Error("No checkpoint has been saved in this browser."));
      request.onerror = () => reject(request.error);
    });
  } finally {
    db.close();
  }
}
