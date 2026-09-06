const database = "fractal-control-recordings";
let connection;
export function openDatabase() {
  connection ||= new Promise((resolve, reject) => {
    const request = indexedDB.open(database, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      db.createObjectStore("runs", { keyPath: "id" });
      db.createObjectStore("records", { keyPath: ["run", "kind", "index"] });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  }).catch((error) => {
    connection = undefined;
    throw error;
  });
  return connection;
}
export async function transaction(stores, mode, operation) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(stores, mode);
    let result;
    tx.oncomplete = () => resolve(result);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () =>
      reject(tx.error || new Error("Recording transaction aborted"));
    try {
      result = operation(tx);
    } catch (error) {
      tx.abort();
      reject(error);
    }
  });
}
export async function read(store, key) {
  let result;
  await transaction([store], "readonly", (tx) => {
    const r = tx.objectStore(store).get(key);
    r.onsuccess = () => (result = r.result);
  });
  return result;
}
export async function listRuns() {
  let rows;
  await transaction(["runs"], "readonly", (tx) => {
    const r = tx.objectStore("runs").getAll();
    r.onsuccess = () => (rows = r.result);
  });
  return rows.sort((a, b) => b.updated - a.updated);
}
export async function records(run) {
  let rows;
  await transaction(["records"], "readonly", (tx) => {
    const r = tx
      .objectStore("records")
      .getAll(
        IDBKeyRange.bound(
          [run, "", 0],
          [run, "\uffff", Number.MAX_SAFE_INTEGER],
        ),
      );
    r.onsuccess = () => (rows = r.result);
  });
  return rows;
}
export async function deleteRun(run) {
  await transaction(["runs", "records"], "readwrite", (tx) => {
    tx.objectStore("runs").delete(run);
    tx.objectStore("records").delete(
      IDBKeyRange.bound([run, "", 0], [run, "\uffff", Number.MAX_SAFE_INTEGER]),
    );
  });
}
