//! Na GPU: upload RAM→VRAM, kernel lendo o slot emprestado, depois evicção e upload de outro expert no mesmo slot.
use cutile::prelude::*;
use p1_posse::kernels::read_expert;
use p1_posse::{HostExpert, SlotCache};

const N: usize = 1024;

fn read(cache: &SlotCache, i: usize, stream: &Arc<cuda_core::Stream>) -> Vec<f32> {
    let y = api::zeros::<f32>(&[N]).sync_on(stream).unwrap().partition([128]);
    read_expert(y, cache.lease(i)).first().unpartition().to_host_vec().sync_on(stream).unwrap()
}

#[test]
fn upload_then_kernel_reads_the_slot_and_a_new_expert_replaces_it() {
    let device = Device::new(0).unwrap();
    let stream = device.new_stream().unwrap();
    let mut cache = SlotCache::new(2, N, &stream).unwrap();
    let a = HostExpert(Arc::new((0..N).map(|i| i as f32).collect()));
    let b = HostExpert(Arc::new((0..N).map(|i| -(i as f32) * 0.5).collect()));

    cache.upload(0, &a).sync_on(&stream).unwrap();
    assert_eq!(read(&cache, 0, &stream), *a.0);

    // O kernel já terminou (sync): agora o slot pode ser sobrescrito.
    cache.upload(0, &b).sync_on(&stream).unwrap();
    assert_eq!(read(&cache, 0, &stream), *b.0);

    cache.evict(1);
}
