import { metadata } from "./registry.js";
import { createRunModel } from "./run-model.js";
export const demos=metadata.filter(d=>d.part==="V").map(d=>({...d,create:args=>createRunModel({...args,id:d.id})}));
