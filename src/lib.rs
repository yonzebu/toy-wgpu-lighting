#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod app;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    app::run().await;
}
