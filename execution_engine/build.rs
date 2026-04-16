fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_path = "../proto/trading.proto";
    let proto_dir = "../proto";

    println!("cargo:rerun-if-changed={}", proto_path);

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&[proto_path], &[proto_dir])?;

    Ok(())
}
