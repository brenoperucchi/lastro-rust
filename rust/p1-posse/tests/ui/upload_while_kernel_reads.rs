use cutile::prelude::*;
use p1_posse::kernels::read_expert;
use p1_posse::{HostExpert, SlotCache};

fn main() {
    let device = Device::new(0).unwrap();
    let stream = device.new_stream().unwrap();
    let mut cache = SlotCache::new(1, 1024, &stream).unwrap();
    let other = HostExpert(Arc::new(vec![1.0; 1024]));
    let y = api::zeros::<f32>(&[1024]).sync_on(&stream).unwrap().partition([128]);
    let op = read_expert(y, cache.lease(0));
    let _ = cache.upload(0, &other).sync_on(&stream); // sobrescreve o slot que o kernel lê
    let _ = op.first().unpartition().to_host_vec().sync_on(&stream);
}
