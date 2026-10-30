use cutile::prelude::*;
use p1_posse::kernels::read_expert;
use p1_posse::{HostExpert, SlotCache};

fn main() {
    let device = Device::new(0).unwrap();
    let stream = device.new_stream().unwrap();
    let mut cache = SlotCache::new(1, 1024, &stream).unwrap();
    let e = HostExpert(Arc::new(vec![1.0; 1024]));
    let upload = cache.upload(0, &e);
    let y = api::zeros::<f32>(&[1024]).sync_on(&stream).unwrap().partition([128]);
    let _ = read_expert(y, cache.lease(0)).first().unpartition().to_host_vec().sync_on(&stream); // lê antes da cópia
    let _ = upload.sync_on(&stream);
}
