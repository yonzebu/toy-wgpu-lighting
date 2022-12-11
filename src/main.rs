mod app;

fn main() {
    // println!("{:?}", std::env::current_dir().unwrap());
    pollster::block_on(app::run());
    // if std::panic::catch_unwind(|| pollster::block_on(app::run())).is_ok() {
    //     println!("finished???");
    // } else {
    //     println!("panicked, oh no!!!");
    // }
    // loop {}
}
