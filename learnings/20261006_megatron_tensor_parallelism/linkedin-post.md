# LinkedIn post: Megatron-style tensor parallelism

**Attach:** `tp-page1.png` … `tp-page5.png` as a carousel (in order).

---

Lately, I've been going through how Megatron-LM splits a transformer layer across GPUs.

Tensor parallelism splits one matmul across GPUs: split the weight by columns (column parallel) and each GPU gets a slice of the output, or split it by rows (row parallel) and each GPU gets a partial sum.

The trick is to pair them: column parallel first, then row parallel, so the output in between stays sharded and the only communication is one all-reduce at the end.

In plain English, an MLP block is linear → GeLU → linear, and since GeLU works on each element on its own, each GPU can run its half of the block without talking to the other.

The order matters, though: split the first matmul by rows and you get partial sums, and GeLU(a + b) isn't GeLU(a) + GeLU(b), so you'd need an extra sync before the activation.

I drew it out in 5 pages and checked it in NumPy. Notes and code are in my kaizen repo (link in comments).

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261006_megatron_tensor_parallelism
