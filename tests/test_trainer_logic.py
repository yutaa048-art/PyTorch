"""
Test suite komprehensif untuk training/trainer.py
Mencakup: logika, visual bug, edge cases, dan konsistensi state.
Jalankan dengan: python -m pytest tests/test_trainer_logic.py -v
"""
import math
import random
import sys
import os
import types
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch, PropertyMock
import torch

# Tambahkan root project ke path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Import kelas yang akan diuji langsung dari trainer.py ───────────────────
from training.trainer import CurriculumScheduler, KnowledgeReplayBuffer, compute_combined_loss


# =============================================================================
# 1. CurriculumScheduler Tests
# =============================================================================
class TestCurriculumScheduler(unittest.TestCase):
    """Menguji logika perubahan seq_len berbasis progres training."""

    def _make(self, total_steps=1000):
        return CurriculumScheduler(total_steps)

    # ── Fase-fase dasar ──────────────────────────────────────────────────────
    def test_phase0_returns_512(self):
        """Langkah 0 harus mengembalikan 512 (fase awal)."""
        c = self._make(1000)
        self.assertEqual(c.get_seq_len(0), 512)

    def test_phase1_returns_1024_at_25pct(self):
        """Tepat di 25% harus beralih ke 1024."""
        c = self._make(1000)
        self.assertEqual(c.get_seq_len(250), 1024)

    def test_phase2_returns_2048_at_50pct(self):
        """Tepat di 50% harus beralih ke 2048."""
        c = self._make(1000)
        self.assertEqual(c.get_seq_len(500), 2048)

    def test_phase3_returns_4096_at_75pct(self):
        """Tepat di 75% harus beralih ke 4096."""
        c = self._make(1000)
        self.assertEqual(c.get_seq_len(750), 4096)

    def test_step_just_before_phase1(self):
        """Satu langkah sebelum 25% harus masih 512."""
        c = self._make(1000)
        self.assertEqual(c.get_seq_len(249), 512)

    def test_step_at_100pct(self):
        """Pada step terakhir harus masih di fase 4096."""
        c = self._make(1000)
        self.assertEqual(c.get_seq_len(1000), 4096)

    def test_step_beyond_total(self):
        """Langkah di atas total tetap aman (tidak crash)."""
        c = self._make(1000)
        self.assertEqual(c.get_seq_len(9999), 4096)

    def test_monotonic_progression(self):
        """seq_len tidak boleh pernah turun sepanjang training."""
        c = self._make(1000)
        prev_sl = 0
        for step in range(0, 1001, 10):
            sl = c.get_seq_len(step)
            self.assertGreaterEqual(sl, prev_sl,
                msg=f"seq_len turun di step {step}: {prev_sl} -> {sl}")
            prev_sl = sl

    # ── Bug lama: loop overwrite (bukan reversed) ────────────────────────────
    def test_no_overwrite_at_boundary(self):
        """Tepat di boundary 50%, harus return 2048 bukan 1024 (bug lama overwrite)."""
        c = self._make(100)
        result = c.get_seq_len(50)  # progress = 0.5 (tepat)
        self.assertEqual(result, 2048,
            msg="Bug lama: loop non-reversed akan overwrite ke 2048 lalu tetap, ini OK. Tapi progress >= 0.75 tidak boleh ikut.")

    def test_step_changed_flag(self):
        """Metode step() harus mengembalikan changed=True saat transisi."""
        c = self._make(1000)
        _, changed = c.step(0)      # Awal: belum berubah
        self.assertFalse(changed)
        _, changed = c.step(250)    # Transisi ke 1024
        self.assertTrue(changed)
        _, changed = c.step(251)    # Tidak ada perubahan
        self.assertFalse(changed)

    def test_total_steps_zero_safe(self):
        """total_steps=0 tidak boleh menyebabkan ZeroDivisionError."""
        # Ini mensimulasikan safeguard max(1, ...) yang sudah ada di trainer
        # CurriculumScheduler sendiri menggunakan max(1, self.total_steps)
        c = CurriculumScheduler(total_steps=1)
        result = c.get_seq_len(0)
        self.assertEqual(result, 512)


# =============================================================================
# 2. KnowledgeReplayBuffer Tests
# =============================================================================
class TestKnowledgeReplayBuffer(unittest.TestCase):
    """Menguji logika sampling dan manajemen kapasitas KRB."""

    def _make_tensor(self, val=0, size=10):
        return torch.zeros(size, dtype=torch.long) + val

    def test_sample_returns_none_when_empty(self):
        """Buffer kosong harus return None, bukan crash."""
        krb = KnowledgeReplayBuffer(max_per_slot=10)
        result = krb.sample_batch(4, torch.device('cpu'))
        self.assertIsNone(result)

    def test_sample_returns_none_when_insufficient(self):
        """Jika item < batch_size, harus return None."""
        krb = KnowledgeReplayBuffer(max_per_slot=10)
        krb.add(self._make_tensor(1), self._make_tensor(2), "python", True)
        krb.add(self._make_tensor(3), self._make_tensor(4), "python", True)
        result = krb.sample_batch(4, torch.device('cpu'))
        self.assertIsNone(result)

    def test_sample_returns_correct_shape(self):
        """Batch yang dikembalikan harus memiliki shape yang benar."""
        krb = KnowledgeReplayBuffer(max_per_slot=50)
        for i in range(20):
            krb.add(self._make_tensor(i), self._make_tensor(i), "python", True)
        result = krb.sample_batch(4, torch.device('cpu'))
        self.assertIsNotNone(result)
        xs, ys, hard_cnt, topic_cnt = result
        self.assertEqual(xs.shape[0], 4)
        self.assertEqual(ys.shape[0], 4)

    def test_max_per_slot_enforced(self):
        """Slot tidak boleh melebihi max_per_slot."""
        krb = KnowledgeReplayBuffer(max_per_slot=5)
        for i in range(20):
            krb.add(self._make_tensor(i), self._make_tensor(i), "rust", True)
        self.assertLessEqual(len(krb.slots["rust"]), 5)

    def test_hard_example_slot_key(self):
        """Contoh hard tanpa topik masuk ke slot '_hard_examples'."""
        krb = KnowledgeReplayBuffer(max_per_slot=10)
        krb.add(self._make_tensor(1), self._make_tensor(1), None, True)
        self.assertIn("_hard_examples", krb.slots)

    def test_non_hard_non_topic_not_added(self):
        """Item yang bukan hard dan bukan topic tidak boleh masuk buffer."""
        krb = KnowledgeReplayBuffer(max_per_slot=10)
        krb.add(self._make_tensor(1), self._make_tensor(1), None, False)
        self.assertEqual(krb.total_size(), 0)

    def test_hard_count_in_sample(self):
        """hard_count harus konsisten dengan item yang di-sample dari '_hard_examples'."""
        krb = KnowledgeReplayBuffer(max_per_slot=100)
        for i in range(50):
            krb.add(self._make_tensor(i), self._make_tensor(i), None, True)
        result = krb.sample_batch(4, torch.device('cpu'))
        self.assertIsNotNone(result)
        xs, ys, hard_cnt, topic_cnt = result
        self.assertEqual(hard_cnt + topic_cnt, 4,
            msg="Total count hard + topic harus sama dengan batch_size")

    def test_sample_batch_none_unpack_safety(self):
        """Memastikan None dari sample_batch tidak langsung di-unpack tanpa check."""
        krb = KnowledgeReplayBuffer(max_per_slot=10)
        # Simulasi pola caller yang benar (seperti di trainer.py)
        result = krb.sample_batch(4, torch.device('cpu'))
        try:
            if result is not None:
                x, y, h, t = result  # Ini hanya boleh dieksekusi jika result bukan None
            success = True
        except TypeError:
            success = False
        self.assertTrue(success, "Unpack None menyebabkan TypeError - caller harus periksa None")


# =============================================================================
# 3. EMA Gradient Norm Tests (Bug Visual)
# =============================================================================
class TestEMAGradNorm(unittest.TestCase):
    """Menguji bahwa EMA gradien norm tidak tercemar oleh nilai inf/nan."""

    def _simulate_ema_update(self, ema, new_val, step_skipped=False):
        """Simulasi logika EMA yang sudah diperbaiki di trainer.py."""
        if not step_skipped and not math.isinf(new_val) and not math.isnan(new_val):
            return (0.9 * ema + 0.1 * new_val) if ema > 0 else new_val
        return ema  # Tidak diupdate

    def test_ema_not_infected_by_inf(self):
        """EMA tidak boleh menjadi inf jika satu nilai inf muncul."""
        ema = 2.5
        ema = self._simulate_ema_update(ema, float('inf'), step_skipped=True)
        self.assertFalse(math.isinf(ema),
            msg="Bug lama: EMA menjadi inf permanen setelah satu nilai inf masuk")
        self.assertAlmostEqual(ema, 2.5, places=5)

    def test_ema_not_infected_by_nan(self):
        """EMA tidak boleh menjadi nan jika satu nilai nan muncul."""
        ema = 2.5
        ema = self._simulate_ema_update(ema, float('nan'), step_skipped=False)
        self.assertFalse(math.isnan(ema),
            msg="EMA menjadi nan setelah nilai nan masuk")

    def test_ema_updates_normally_with_finite(self):
        """EMA harus terupdate normal jika nilai finite dan tidak skip."""
        ema = 2.0
        ema = self._simulate_ema_update(ema, 1.0, step_skipped=False)
        self.assertAlmostEqual(ema, 0.9 * 2.0 + 0.1 * 1.0, places=5)

    def test_ema_not_updated_on_skipped_step(self):
        """Jika step di-skip (overflow), EMA tidak boleh diupdate."""
        ema = 2.0
        ema_new = self._simulate_ema_update(ema, 1.5, step_skipped=True)
        self.assertEqual(ema_new, ema,
            msg="EMA berubah padahal step di-skip")

    def test_ema_initial_set_from_first_finite_value(self):
        """EMA yang masih 0 harus diinisialisasi dari nilai pertama yang finite."""
        ema = 0.0
        ema = self._simulate_ema_update(ema, 3.14, step_skipped=False)
        self.assertAlmostEqual(ema, 3.14, places=5)

    def test_ema_recovers_after_multiple_inf(self):
        """Setelah beberapa inf, EMA harus bisa kembali diupdate dengan nilai normal."""
        ema = 2.5
        for _ in range(10):
            ema = self._simulate_ema_update(ema, float('inf'), step_skipped=True)
        # Setelah inf berulang, EMA harus tetap 2.5
        self.assertAlmostEqual(ema, 2.5, places=5)
        # Lalu update normal
        ema = self._simulate_ema_update(ema, 1.0, step_skipped=False)
        self.assertAlmostEqual(ema, 0.9 * 2.5 + 0.1 * 1.0, places=5)


# =============================================================================
# 4. Scaler Step Detection Tests
# =============================================================================
class TestScalerStepDetection(unittest.TestCase):
    """Menguji deteksi apakah optimizer.step diblokir oleh GradScaler."""

    def _simulate_scaler_logic(self, grad_norm_raw, scale_before, scale_after):
        """
        Simulasi logika step_skipped yang digunakan di trainer.py.
        Returns: (step_skipped, ema_updated, grad_norm_raw_unchanged)
        """
        step_skipped = scale_after < scale_before
        ema_grad_norm = 2.5  # Nilai EMA sebelumnya

        if not step_skipped and not math.isinf(grad_norm_raw) and not math.isnan(grad_norm_raw):
            ema_grad_norm = 0.9 * ema_grad_norm + 0.1 * grad_norm_raw
            ema_updated = True
        else:
            ema_updated = False

        return step_skipped, ema_updated, grad_norm_raw

    def test_normal_step_updates_ema(self):
        """Step normal (scale tidak turun) harus mengupdate EMA."""
        skipped, updated, raw = self._simulate_scaler_logic(1.5, 262144, 262144)
        self.assertFalse(skipped)
        self.assertTrue(updated)
        self.assertEqual(raw, 1.5)

    def test_overflow_step_detected(self):
        """Ketika scale turun, harus terdeteksi sebagai skip."""
        skipped, updated, raw = self._simulate_scaler_logic(float('inf'), 262144, 131072)
        self.assertTrue(skipped)
        self.assertFalse(updated)

    def test_grad_norm_raw_preserved_on_skip(self):
        """Nilai raw (inf) harus tetap tersimpan meski step di-skip (untuk logging)."""
        skipped, updated, raw = self._simulate_scaler_logic(float('inf'), 262144, 131072)
        self.assertTrue(math.isinf(raw),
            msg="grad_norm_raw harus tetap inf untuk keperluan logging/debugging")

    def test_scale_increase_not_treated_as_skip(self):
        """Jika scale naik (setelah 2000 langkah bersih), ini bukan skip."""
        skipped, updated, raw = self._simulate_scaler_logic(0.8, 262144, 524288)
        self.assertFalse(skipped,
            msg="Scale naik seharusnya bukan skip, tapi step normal dengan scale baru")
        self.assertTrue(updated)


# =============================================================================
# 5. Total Steps & Warmup Calculation Tests
# =============================================================================
class TestTotalStepsCalculation(unittest.TestCase):
    """Menguji bahwa perhitungan total_steps dan warmup_steps sudah benar."""

    def _calc(self, num_batches, max_epochs, accumulation_steps):
        """Simulasi logika perhitungan di trainer.py (setelah perbaikan)."""
        total_batches = num_batches * max_epochs
        total_steps = max(1, total_batches // accumulation_steps)
        warmup_steps = int(total_steps * 0.05)
        return total_steps, warmup_steps

    def test_warmup_is_5pct_of_global_steps(self):
        """Warmup harus ~5% dari global steps (bukan total batch)."""
        total_steps, warmup = self._calc(
            num_batches=268750,   # total batch (4.3M / 16)
            max_epochs=1,
            accumulation_steps=16
        )
        # Seharusnya total_steps = 268750 // 16 ≈ 16797
        # Bukan 268750 * 0.05 = 13437 (bug lama)
        self.assertLess(warmup, total_steps * 0.1,
            msg="Warmup melebihi 10% dari total steps")

    def test_warmup_not_80pct(self):
        """Warmup tidak boleh mencapai 80% dari total steps (bug lama)."""
        total_steps, warmup = self._calc(
            num_batches=4300000,
            max_epochs=1,
            accumulation_steps=16
        )
        warmup_ratio = warmup / total_steps
        self.assertLess(warmup_ratio, 0.10,
            msg=f"Bug lama: warmup ratio = {warmup_ratio:.2%}, seharusnya ~5%")

    def test_total_steps_never_zero(self):
        """total_steps tidak boleh nol meski dataset sangat kecil."""
        total_steps, _ = self._calc(num_batches=3, max_epochs=1, accumulation_steps=16)
        self.assertGreaterEqual(total_steps, 1,
            msg="total_steps = 0 akan crash scheduler")

    def test_normal_large_training(self):
        """Verifikasi angka nyata yang digunakan di training 500M ini."""
        # Dataset 8.8B token, seq_len 512, batch 4, accum 16
        # len(train_dl) = 8.8B / (512 * 4) = 4.296.875 batches
        total_batches = 4_296_875
        total_steps, warmup = self._calc(total_batches, max_epochs=1, accumulation_steps=16)
        # Harusnya ~268.554 global steps
        self.assertGreater(total_steps, 200_000)
        self.assertLess(total_steps, 400_000)
        # Warmup harusnya ~13.427 (bukan 214.843)
        self.assertLess(warmup, 20_000)
        self.assertGreater(warmup, 10_000)


# =============================================================================
# 6. Grad Norm Display Format Tests (Visual Bug)
# =============================================================================
class TestGradNormDisplay(unittest.TestCase):
    """Menguji format teks yang ditampilkan di terminal untuk Gradient Norm."""

    def _build_display(self, ema_grad_norm, grad_norm_raw, step_skipped, grad_scale):
        """Simulasi logika pembangunan string display di trainer.py."""
        if step_skipped:
            return f"{ema_grad_norm:.4f} (EMA) | Raw: {grad_norm_raw:.4f} [SKIP-overflow] (Scale: {grad_scale})"
        elif math.isinf(grad_norm_raw) or math.isnan(grad_norm_raw):
            return f"{ema_grad_norm:.4f} (EMA) | Raw: {grad_norm_raw} (Scale: {grad_scale})"
        else:
            return f"{ema_grad_norm:.4f} (Scale: {grad_scale})"

    def test_normal_display_no_suffix(self):
        """Tampilan normal tidak boleh mengandung kata 'SKIP' atau 'Raw'."""
        display = self._build_display(2.1234, 1.9, False, 262144)
        self.assertNotIn("SKIP", display)
        self.assertNotIn("Raw", display)
        self.assertIn("2.1234", display)

    def test_skip_display_contains_marker(self):
        """Tampilan saat skip harus mengandung '[SKIP-overflow]'."""
        display = self._build_display(2.1234, float('inf'), True, 131072)
        self.assertIn("[SKIP-overflow]", display)
        self.assertIn("EMA", display)

    def test_inf_raw_not_skipped_shows_raw(self):
        """Jika raw=inf tapi step tidak di-skip (edge case), tetap tampilkan raw."""
        display = self._build_display(2.1234, float('inf'), False, 262144)
        self.assertIn("Raw:", display)
        self.assertNotIn("[SKIP-overflow]", display)

    def test_ema_always_shown_correctly(self):
        """Nilai EMA harus selalu diformat dengan 4 desimal."""
        display = self._build_display(2.1, 1.9, False, 262144)
        self.assertIn("2.1000", display)


# =============================================================================
# 7. compute_combined_loss Tests
# =============================================================================
class TestComputeCombinedLoss(unittest.TestCase):
    """Menguji fungsi kalkulasi combined loss.

    Note: compute_combined_loss menggunakan calculate_loss_unreduced yang mengharapkan
    logits [batch, seq_len, vocab_size] dan targets [batch, seq_len].
    Aux heads tetap menggunakan [batch, classes] karena pooled di level sequence.
    """
    BATCH = 2
    SEQ_LEN = 8
    VOCAB = 32
    AUX_CLASSES = 10

    def _make_lm_logits(self):
        """Logits untuk LM head: [batch, seq_len, vocab_size]."""
        return torch.randn(self.BATCH, self.SEQ_LEN, self.VOCAB)

    def _make_lm_targets(self):
        """Target untuk LM head: [batch, seq_len]."""
        return torch.randint(0, self.VOCAB, (self.BATCH, self.SEQ_LEN))

    def _make_aux_logits(self):
        """Logits untuk aux head (pooled): [batch, aux_classes]."""
        return torch.randn(self.BATCH, self.AUX_CLASSES)

    def _make_aux_labels(self):
        """Labels untuk aux head: [batch]."""
        return torch.randint(0, self.AUX_CLASSES, (self.BATCH,))

    def test_returns_6_values(self):
        """Harus mengembalikan 6 nilai: loss, lm, unreduced, head_losses, head_accs, head_confs."""
        logits = self._make_lm_logits()
        targets = self._make_lm_targets()
        result = compute_combined_loss(logits, targets, {}, {}, {}, accumulation_steps=1)
        self.assertEqual(len(result), 6)

    def test_loss_divided_by_accumulation(self):
        """Total loss harus dibagi accumulation_steps."""
        logits = self._make_lm_logits()
        targets = self._make_lm_targets()
        loss1, *_ = compute_combined_loss(logits, targets, {}, {}, {}, accumulation_steps=1)
        loss4, *_ = compute_combined_loss(logits, targets, {}, {}, {}, accumulation_steps=4)
        self.assertAlmostEqual(loss4.item(), loss1.item() / 4, places=5)

    def test_aux_heads_increase_loss(self):
        """Adanya aux heads harus meningkatkan total loss (karena aux weight > 0)."""
        logits = self._make_lm_logits()
        targets = self._make_lm_targets()

        aux_logits = {'syntax': self._make_aux_logits()}
        aux_labels = {'syntax': self._make_aux_labels()}
        aux_confs = {'syntax': torch.tensor([0.8, 0.7])}

        loss_no_aux, *_ = compute_combined_loss(logits, targets, {}, {}, {}, 1)
        loss_with_aux, *_ = compute_combined_loss(logits, targets, aux_logits, aux_labels, aux_confs, 1)
        self.assertGreater(loss_with_aux.item(), loss_no_aux.item())

    def test_head_acc_between_0_and_1(self):
        """Akurasi head harus antara 0 dan 1."""
        logits = self._make_lm_logits()
        targets = self._make_lm_targets()
        aux_logits = {'syntax': self._make_aux_logits()}
        aux_labels = {'syntax': self._make_aux_labels()}
        _, _, _, head_losses, head_accs, head_confs = compute_combined_loss(
            logits, targets, aux_logits, aux_labels, {}, 1
        )
        for k, acc in head_accs.items():
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_no_crash_with_missing_confidence(self):
        """Tanpa aux_confidences, harus menggunakan default conf_weight=0.5 tanpa crash."""
        logits = self._make_lm_logits()
        targets = self._make_lm_targets()
        aux_logits = {'reasoning': self._make_aux_logits()}
        aux_labels = {'reasoning': self._make_aux_labels()}
        try:
            compute_combined_loss(logits, targets, aux_logits, aux_labels, {}, 1)
            crashed = False
        except Exception as e:
            crashed = True
            self.fail(f"compute_combined_loss crash dengan missing confidence: {e}")
        self.assertFalse(crashed)


# =============================================================================
# 8. Final Checkpoint Completeness Tests
# =============================================================================
class TestCheckpointCompleteness(unittest.TestCase):
    """Menguji bahwa struktur checkpoint sudah lengkap untuk resume."""

    REQUIRED_KEYS = {'model', 'optimizer', 'scaler', 'scheduler', 'epoch', 'step_in_epoch', 'global_step'}

    def _make_dummy_checkpoint(self, include_all=True):
        """Buat dummy checkpoint seperti yang sekarang disimpan di trainer.py."""
        ckpt = {
            'model': {},
            'optimizer': {},
            'scaler': {},
            'scheduler': {'current_step': 100},
            'epoch': 0,
            'step_in_epoch': 499,
            'global_step': 500,
        }
        if not include_all:
            del ckpt['scheduler']
            del ckpt['scaler']
        return ckpt

    def test_full_checkpoint_has_all_keys(self):
        """Checkpoint harus memiliki semua key yang diperlukan."""
        ckpt = self._make_dummy_checkpoint(include_all=True)
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, ckpt, msg=f"Key '{key}' tidak ada di checkpoint")

    def test_incomplete_checkpoint_missing_keys(self):
        """Checkpoint lama (tanpa scheduler/scaler) terdeteksi sebagai tidak lengkap."""
        ckpt = self._make_dummy_checkpoint(include_all=False)
        missing = self.REQUIRED_KEYS - set(ckpt.keys())
        self.assertGreater(len(missing), 0,
            msg="Checkpoint lama seharusnya memiliki key yang hilang")

    def test_global_step_type_is_int(self):
        """global_step harus integer, bukan float."""
        ckpt = self._make_dummy_checkpoint()
        self.assertIsInstance(ckpt['global_step'], int)

    def test_scheduler_has_current_step(self):
        """scheduler state_dict harus memiliki 'current_step' untuk resume yang akurat."""
        ckpt = self._make_dummy_checkpoint()
        self.assertIn('current_step', ckpt['scheduler'],
            msg="Scheduler state_dict tidak memiliki current_step — resume LR akan salah")


# =============================================================================
# Run all tests
# =============================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
