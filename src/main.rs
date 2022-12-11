#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod app;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    app::run().await;
}

// #[cfg(not(target_arch = "wasm32"))]
fn main() {
    use cfg_if::cfg_if;

    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {

        } else {
            pollster::block_on(app::run());
        }
    }
    // println!("{:?}", std::env::current_dir().unwrap());
    // if std::panic::catch_unwind(|| pollster::block_on(app::run())).is_ok() {
    //     println!("finished???");
    // } else {
    //     println!("panicked, oh no!!!");
    // }
    // loop {}
}
