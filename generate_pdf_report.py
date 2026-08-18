"""
generate_pdf_report.py
======================
Generates an exhaustive, beautifully styled PDF project report for the
Real-Time EMG Gesture Recognition & IoT Control System using ReportLab.
"""

import os, sys, json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas

# Define custom NumberedCanvas for professional "Page X of Y" footers and running headers
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        if self._pageNumber > 1:
            # Running Header
            self.drawString(54, 11 * inch - 36, "Real-Time EMG Gesture Recognition & IoT Control System — Project Report")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.75)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

            # Running Footer
            self.line(54, 48, 8.5 * inch - 54, 48)
            self.drawString(54, 34, "Confidential — Engineering & Research Documentation")
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(8.5 * inch - 54, 34, page_text)
        else:
            # First page bottom footer
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.75)
            self.line(54, 48, 8.5 * inch - 54, 48)
            self.drawString(54, 34, "EMG Gesture Recognition System | IoT Major Project")
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(8.5 * inch - 54, 34, page_text)

        self.restoreState()


def build_pdf(filename="EMG_Gesture_Recognition_Project_Report.pdf"):
    # Target 8.5 x 11 inch with 54pt (0.75 in) margins
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    c_primary   = colors.HexColor("#1e3a8a")   # Deep Navy
    c_secondary = colors.HexColor("#2563eb")   # Vibrant Blue
    c_accent    = colors.HexColor("#7c3aed")   # Purple
    c_dark      = colors.HexColor("#0f172a")   # Charcoal
    c_muted     = colors.HexColor("#475569")   # Slate
    c_light     = colors.HexColor("#f8fafc")   # Off-white
    c_border    = colors.HexColor("#cbd5e1")   # Light gray border
    c_tag_bg    = colors.HexColor("#eff6ff")   # Soft blue

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=c_primary,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11.5,
        leading=15,
        textColor=c_muted,
        spaceAfter=14
    )

    meta_badge_style = ParagraphStyle(
        'MetaBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=c_secondary
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=6
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b")
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=c_dark
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=c_primary
    )

    table_cell_header = ParagraphStyle(
        'TableHeader',
        parent=table_cell,
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.white
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # ─────────────────────────────────────────────────────────────
    # COVER / HEADER BLOCK
    # ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("Real-Time EMG Gesture Recognition & IoT Control System", title_style))
    story.append(Paragraph("<b>Comprehensive Technical Analysis Report:</b> Biosignal Synthesis, 15-Feature Engineering, Classical ML Ensembles, Deep Learning (DTSF-CNN), Live React Telemetry & Smart Appliance Automation", subtitle_style))
    
    # Metadata bar table
    meta_data = [
        [
            Paragraph("<b>Framework:</b> PyTorch 2.x & Scikit-Learn", meta_badge_style),
            Paragraph("<b>Frontend:</b> React 18 + Vite 5", meta_badge_style),
            Paragraph("<b>Edge Target:</b> RP2040 (Pico W)", meta_badge_style),
            Paragraph("<b>Sampling Rate:</b> 500 Hz (256-sample window)", meta_badge_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[125, 125, 125, 129])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_tag_bg),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#bfdbfe")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dbeafe")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=4, spaceAfter=10))

    # ─────────────────────────────────────────────────────────────
    # 1. EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", h1_style))
    exec_text = (
        "Electromyography (EMG) measures biopotential voltages generated by skeletal muscle fibers during contraction. "
        "This project presents an integrated cyber-physical platform capable of synthesizing realistic surface EMG signals, "
        "extracting a 15-dimensional feature vector, classifying multi-class gestures using machine learning and deep learning, "
        "and driving smart IoT appliances in real time with dynamic per-user calibration and rejection thresholding."
    )
    story.append(Paragraph(exec_text, body_style))

    # Summary Highlight Box
    highlight_data = [[
        Paragraph(
            "<b>Key Highlights & Benchmarks:</b><br/>"
            "• <b>Deep Learning DTSF-CNN:</b> 258,034 parameter Dual-Path Temporal-Spectral Fusion CNN achieving <b>56.32% ± 0.80% 5-fold CV accuracy</b>.<br/>"
            "• <b>Tuned Random Forest Ensemble:</b> 200 trees achieving <b>68.43% test accuracy</b> on unseen subjects.<br/>"
            "• <b>Real-Time Streaming Latency:</b> &lt; 0.20 ms forward pass (&lt; 5 ms real-time streaming constraint satisfied).<br/>"
            "• <b>Microcontroller Ready:</b> Automatic export of priors, Gaussian means, and covariances for microcontrollers (Raspberry Pi Pico W).",
            callout_style
        )
    ]]
    highlight_table = Table(highlight_data, colWidths=[504])
    highlight_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('LINELEFT', (0,0), (-1,-1), 3.5, c_secondary),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(highlight_table)
    story.append(Spacer(1, 8))

    # ─────────────────────────────────────────────────────────────
    # 2. PHYSIOLOGICAL SIGNAL GENERATION & SIMULATION ENGINE
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("2. Physiological EMG Signal Modeling & Simulation", h1_style))
    story.append(Paragraph(
        "The pipeline includes a realistic <b>multi-component mathematical EMG synthesis model</b> operating at <b>500 Hz</b> "
        "with <b>256-sample (~512 ms)</b> windows:",
        body_style
    ))

    story.append(Paragraph(
        "EMG(t) = [ A * S_user * (1 - 0.2*fatigue) * (1 + 0.14*sin(3.6*pi*t) + 0.06*sin(0.8*pi*t)) * N(0,1) ] + 0.07*A*sin(2*pi*fc*t) + Noise + DC",
        code_style
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Where A is the gesture target amplitude, S_user is inter-subject scaling (0.50x to 1.60x), "
        "fatigue in [0, 0.30] is time-dependent muscle force decay, fc is the characteristic burst frequency, "
        "and random motion artifacts (0.35 V impulses) are injected at an 8% occurrence rate.",
        body_style
    ))

    # Table of Gesture Classes
    gesture_table_data = [
        [
            Paragraph("Gesture", table_cell_header),
            Paragraph("Icon", table_cell_header),
            Paragraph("Physiological Action", table_cell_header),
            Paragraph("Amp (V)", table_cell_header),
            Paragraph("Burst Freq", table_cell_header),
            Paragraph("Noise", table_cell_header),
            Paragraph("Exp. RMS", table_cell_header),
        ],
        [Paragraph("<b>FIST</b>", table_cell_bold), Paragraph("Fist", table_cell), Paragraph("Clenched fist contraction", table_cell), Paragraph("0.82", table_cell), Paragraph("155 Hz", table_cell), Paragraph("0.14", table_cell), Paragraph("0.520 V", table_cell)],
        [Paragraph("<b>OPEN_HAND</b>", table_cell_bold), Paragraph("Open", table_cell), Paragraph("Radial finger extension", table_cell), Paragraph("0.50", table_cell), Paragraph("102 Hz", table_cell), Paragraph("0.10", table_cell), Paragraph("0.310 V", table_cell)],
        [Paragraph("<b>WRIST_UP</b>", table_cell_bold), Paragraph("Up", table_cell), Paragraph("Wrist dorsiflexion", table_cell), Paragraph("0.67", table_cell), Paragraph("128 Hz", table_cell), Paragraph("0.12", table_cell), Paragraph("0.420 V", table_cell)],
        [Paragraph("<b>WRIST_DOWN</b>", table_cell_bold), Paragraph("Down", table_cell), Paragraph("Wrist palmar flexion", table_cell), Paragraph("0.60", table_cell), Paragraph("113 Hz", table_cell), Paragraph("0.11", table_cell), Paragraph("0.370 V", table_cell)],
        [Paragraph("<b>DOUBLE_FLEX</b>", table_cell_bold), Paragraph("Flex", table_cell), Paragraph("Forearm + wrist co-contraction", table_cell), Paragraph("1.02", table_cell), Paragraph("178 Hz", table_cell), Paragraph("0.20", table_cell), Paragraph("0.670 V", table_cell)],
        [Paragraph("<b>RELAX</b>", table_cell_bold), Paragraph("Relax", table_cell), Paragraph("Resting muscular baseline", table_cell), Paragraph("0.04", table_cell), Paragraph("28 Hz", table_cell), Paragraph("0.02", table_cell), Paragraph("0.040 V", table_cell)],
    ]
    gt_table = Table(gesture_table_data, colWidths=[70, 34, 180, 50, 60, 45, 65])
    gt_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(gt_table)
    story.append(Spacer(1, 10))

    # ─────────────────────────────────────────────────────────────
    # 3. 15-DIMENSIONAL FEATURE EXTRACTION PIPELINE
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("3. 15-Dimensional Time-Domain Feature Pipeline", h1_style))
    story.append(Paragraph(
        "A total of 15 handcrafted time-domain, morphology, and frequency-domain surrogate features are extracted per window:",
        body_style
    ))

    feat_table_data = [
        [
            Paragraph("Category", table_cell_header),
            Paragraph("Feature", table_cell_header),
            Paragraph("Mathematical Definition / Principle", table_cell_header),
            Paragraph("Diagnostic Role", table_cell_header),
        ],
        [
            Paragraph("<b>Amplitude & Energy</b>", table_cell_bold),
            Paragraph("<b>MAV, MMAV<br/>RMS, VAR<br/>STD, IEMG</b>", table_cell),
            Paragraph(
                "• MAV = (1/N) * sum(|x_i|)<br/>"
                "• MMAV = (1/N) * sum(w_i * |x_i|) (weighted center 50%)<br/>"
                "• RMS = sqrt((1/N)*sum(x_i^2)), VAR = sigma^2, IEMG = sum(|x_i|)",
                table_cell
            ),
            Paragraph("Primary indicators of muscular recruitment strength and contraction force.", table_cell)
        ],
        [
            Paragraph("<b>Waveform Complexity</b>", table_cell_bold),
            Paragraph("<b>WL<br/>AAC<br/>DASDV</b>", table_cell),
            Paragraph(
                "• WL = sum(|x_i - x_{i-1}|) (cumulative length)<br/>"
                "• AAC = (1/(N-1)) * sum(|x_i - x_{i-1}|)<br/>"
                "• DASDV = sqrt((1/(N-1)) * sum((x_i - x_{i-1})^2))",
                table_cell
            ),
            Paragraph("Quantifies continuous waveform excursion, roughness, and dynamic variance.", table_cell)
        ],
        [
            Paragraph("<b>Frequency Surrogates</b>", table_cell_bold),
            Paragraph("<b>ZC<br/>SSC</b>", table_cell),
            Paragraph(
                "• ZC: Count of zero-crossings with threshold eps = 0.01 V<br/>"
                "• SSC: Count of slope sign changes with eps = 0.003 V",
                table_cell
            ),
            Paragraph("Estimates motor unit firing rate shifts without costly Fourier transforms.", table_cell)
        ],
        [
            Paragraph("<b>Hjorth Parameters</b>", table_cell_bold),
            Paragraph("<b>Activity<br/>Mobility<br/>Complexity</b>", table_cell),
            Paragraph(
                "• Activity = VAR(x(t))<br/>"
                "• Mobility = sqrt(VAR(x') / VAR(x)) (mean frequency)<br/>"
                "• Complexity = Mobility(x') / Mobility(x) (spectral width)",
                table_cell
            ),
            Paragraph("Compact spectral shape parameters sensitive to muscle fatigue and bandwidth.", table_cell)
        ],
        [
            Paragraph("<b>Threshold Rate</b>", table_cell_bold),
            Paragraph("<b>MYOP</b>", table_cell),
            Paragraph("• MYOP = (1/N) * sum(I(|x_i| > 3*sigma))", table_cell),
            Paragraph("Percentage of high-energy motor unit action potential spikes.", table_cell)
        ],
    ]
    feat_table = Table(feat_table_data, colWidths=[90, 80, 204, 130])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 10))

    # ─────────────────────────────────────────────────────────────
    # 4. MACHINE LEARNING ENSEMBLE & BENCHMARKS
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("4. Classical Machine Learning Ensemble & Benchmark Results", h1_style))
    story.append(Paragraph(
        "Evaluated across <b>20 simulated subjects</b> (7,200 total samples) under a subject-independent evaluation protocol "
        "(17 training subjects = 6,120 samples; 3 unseen test subjects = 1,080 samples):",
        body_style
    ))

    ml_res_data = [
        [
            Paragraph("Model Architecture", table_cell_header),
            Paragraph("Model Family", table_cell_header),
            Paragraph("Hyperparameters / Trees", table_cell_header),
            Paragraph("5-Fold CV Accuracy", table_cell_header),
            Paragraph("Test Acc (Unseen)", table_cell_header),
        ],
        [Paragraph("<b>DTSF-CNN (Ours)</b>", table_cell_bold), Paragraph("Deep Learning (PyTorch)", table_cell), Paragraph("258,034 parameters", table_cell), Paragraph("<b>56.32% ± 0.80%</b>", table_cell), Paragraph("29.26%", table_cell)],
        [Paragraph("<b>Random Forest (Tuned)</b>", table_cell_bold), Paragraph("Ensemble Decision Trees", table_cell), Paragraph("200 trees, depth=10, min_leaf=2", table_cell), Paragraph("51.62% ± 0.91%", table_cell), Paragraph("<b>68.43%</b>", table_cell)],
        [Paragraph("<b>Gradient Boosting</b>", table_cell_bold), Paragraph("Boosted Trees", table_cell), Paragraph("150 trees, max_depth=4, lr=0.1", table_cell), Paragraph("51.13% ± 1.11%", table_cell), Paragraph("—", table_cell)],
        [Paragraph("<b>Weighted Soft Voting</b>", table_cell_bold), Paragraph("Meta-Ensemble", table_cell), Paragraph("RF (0.68) + SVM (0.63) + GNB (0.33)", table_cell), Paragraph("54.95% (Train)", table_cell), Paragraph("<b>50.65%</b>", table_cell)],
        [Paragraph("<b>k-Nearest Neighbors</b>", table_cell_bold), Paragraph("Instance-Based", table_cell), Paragraph("k=7, Euclidean, Standardized", table_cell), Paragraph("48.55% ± 0.82%", table_cell), Paragraph("—", table_cell)],
        [Paragraph("<b>Support Vector Machine</b>", table_cell_bold), Paragraph("Kernel Machine", table_cell), Paragraph("RBF Kernel, C=10, gamma='scale'", table_cell), Paragraph("48.40% ± 0.56%", table_cell), Paragraph("62.50%", table_cell)],
        [Paragraph("<b>Gaussian Naive Bayes</b>", table_cell_bold), Paragraph("Probabilistic Generative", table_cell), Paragraph("Gaussian likelihood (Pico target)", table_cell), Paragraph("40.69% ± 0.90%", table_cell), Paragraph("33.43%", table_cell)],
    ]
    ml_table = Table(ml_res_data, colWidths=[130, 105, 125, 75, 69])
    ml_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(ml_table)
    story.append(Spacer(1, 8))

    # Feature Importance Section
    story.append(Paragraph("<b>Feature Importance Ranking (Random Forest):</b>", h2_style))
    fi_text = (
        "1. <b>Hjorth Activity (11.54%)</b> | 2. <b>MMAV (11.05%)</b> | 3. <b>MAV (10.98%)</b> | 4. <b>RMS (10.52%)</b> | "
        "5. <b>IEMG (9.67%)</b> | 6. <b>VAR (9.27%)</b> | 7. <b>STD (9.18%)</b> | 8. <b>WL (8.25%)</b> | 9. <b>DASDV (6.67%)</b> | "
        "10. <b>AAC (5.94%)</b> | 11. <b>ZC (2.43%)</b> | 12. <b>Hjorth Complexity (1.49%)</b> | 13. <b>Hjorth Mobility (1.35%)</b> | "
        "14. <b>SSC (1.19%)</b> | 15. <b>MYOP (0.47%)</b>."
    )
    story.append(Paragraph(fi_text, body_style))
    story.append(Spacer(1, 10))

    # ─────────────────────────────────────────────────────────────
    # 5. DEEP LEARNING: DUAL-PATH TEMPORAL-SPECTRAL FUSION CNN
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Deep Learning Architecture: DTSF-CNN", h1_style))
    story.append(Paragraph(
        "A novel <b>Dual-Path Temporal-Spectral Fusion CNN (DTSF-CNN)</b> was developed in PyTorch (`train_cnn_model.py`) "
        "with <b>258,034 trainable parameters</b>:",
        body_style
    ))

    arch_bullets = (
        "• <b>Path A (Multi-Scale Temporal Convolutions):</b> Three parallel Conv1D branches with kernel sizes "
        "k=7 (~14 ms, motor unit twitches), k=15 (~30 ms, voluntary contraction onset), and k=31 (~62 ms, sustained force envelopes), "
        "followed by 2x ResBlock1D layers and Global Average Pooling (96-dim).<br/>"
        "• <b>Path B (Spectral Attention Block):</b> Calculates Welch Power Spectral Density (33 frequency bins), processed through Conv1D "
        "layers with Squeeze-and-Excitation (SE) channel recalibration (48-dim).<br/>"
        "• <b>Adaptive Sigmoid Fusion Gate:</b> Dynamically weights representations: h_fused = g * h_temp + (1-g) * h_spec.<br/>"
        "• <b>FiLM Conditioning Layer:</b> Modulates deep representations with the 15 handcrafted features via affine transformation: "
        "h_out = gamma(f) * h + beta(f)."
    )
    story.append(Paragraph(arch_bullets, body_style))
    story.append(Spacer(1, 6))

    # Embed cnn_results.png if it exists
    img_path = "cnn_results.png"
    if os.path.exists(img_path):
        story.append(Paragraph("<b>Figure 1: DTSF-CNN Training Curves & Multi-Class Confusion Matrix</b>", h2_style))
        story.append(Image(img_path, width=6.8 * inch, height=2.8 * inch))
        story.append(Spacer(1, 8))

    # ─────────────────────────────────────────────────────────────
    # 6. REAL-TIME TELEMETRY & IOT AUTOMATION DASHBOARD
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("6. Interactive React Telemetry & Smart Appliance IoT Controller", h1_style))
    story.append(Paragraph(
        "The web application (`EMGDashboard_v5.jsx`) is built with React 18 and Vite 5, running completely client-side "
        "with an in-browser Random Forest and GNB inference engine operating at 60 FPS.",
        body_style
    ))

    iot_data = [
        [
            Paragraph("Smart Appliance", table_cell_header),
            Paragraph("Activation Gesture", table_cell_header),
            Paragraph("Deactivation Gesture", table_cell_header),
            Paragraph("Hardware Interface & Telemetry", table_cell_header),
        ],
        [Paragraph("<b>Smart Light</b>", table_cell_bold), Paragraph("FIST", table_cell), Paragraph("OPEN_HAND", table_cell), Paragraph("Binary relay toggle (Active / Inactive)", table_cell)],
        [Paragraph("<b>Ceiling Fan</b>", table_cell_bold), Paragraph("WRIST_UP", table_cell), Paragraph("WRIST_DOWN", table_cell), Paragraph("PWM Speed controller (0 to 1200 RPM)", table_cell)],
        [Paragraph("<b>Smart Door</b>", table_cell_bold), Paragraph("WRIST_DOWN", table_cell), Paragraph("RELAX", table_cell), Paragraph("Solenoid strike lock / unlock actuator", table_cell)],
        [Paragraph("<b>Stepper Motor</b>", table_cell_bold), Paragraph("DOUBLE_FLEX", table_cell), Paragraph("OPEN_HAND", table_cell), Paragraph("Bi-directional robotic joint drive", table_cell)],
        [Paragraph("<b>Smart TV</b>", table_cell_bold), Paragraph("OPEN_HAND", table_cell), Paragraph("RELAX", table_cell), Paragraph("IR / MQTT Power state toggle", table_cell)],
        [Paragraph("<b>AC Unit</b>", table_cell_bold), Paragraph("DOUBLE_FLEX", table_cell), Paragraph("WRIST_UP", table_cell), Paragraph("Thermostat cooling compressor enable", table_cell)],
    ]
    iot_table = Table(iot_data, colWidths=[100, 110, 110, 184])
    iot_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(iot_table)
    story.append(Spacer(1, 8))

    # Guided Calibration & Temporal Smoother
    story.append(Paragraph(
        "<b>Adaptive Calibration & Temporal Filtering:</b><br/>"
        "• <b>6-Step Guided Calibration Wizard:</b> Measures resting DC noise and active contraction RMS values to compute an individualized "
        "scaling multiplier S_cal = median(RMS_obs / RMS_exp) that normalizes feature vectors before classification.<br/>"
        "• <b>Rolling Temporal Smoother:</b> Applies a 5-frame moving average window over probability distributions with a "
        "<b>0.52 confidence rejection threshold</b>, eliminating momentary state flickering and transitional noise.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # ─────────────────────────────────────────────────────────────
    # 7. SYSTEM INSIGHTS, LIMITATIONS & STRATEGIC ROADMAP
    # ─────────────────────────────────────────────────────────────
    story.append(Paragraph("7. Technical Insights, Limitations & Strategic Roadmap", h1_style))
    
    roadmap_text = (
        "<b>Key Technical Insights:</b><br/>"
        "1. <b>Resting Baseline Isolation:</b> The RELAX state achieves <b>100% precision and recall</b> across both classical and deep learning models.<br/>"
        "2. <b>Single-Channel Crosstalk:</b> Differentiating fine wrist movements (WRIST_UP vs. WRIST_DOWN vs. FIST) from a single EMG electrode channel "
        "is physiologically constrained due to spatial motor unit overlap.<br/>"
        "3. <b>Generalization vs. Domain Knowledge:</b> The hand-crafted Random Forest model exhibited stronger cross-user generalization (68.43%) "
        "than the DTSF-CNN (29.26% on unseen synthetic users without fine-tuning), highlighting the robustness of statistical EMG features.<br/><br/>"
        "<b>Strategic Roadmap for Future Development:</b><br/>"
        "• <b>Multi-Channel Array Integration:</b> Expand from 1-channel to 4-channel sEMG (e.g. Myo Armband style) to capture distinct forearm muscle bellies.<br/>"
        "• <b>Sequence Deep Learning:</b> Introduce <b>CNN-BiLSTM</b> or <b>InceptionTime</b> networks to model temporal transitions across contraction phases.<br/>"
        "• <b>TinyML Embedded Quantization:</b> Convert models to 8-bit quantized ONNX / TFLite Micro for sub-10ms direct onboard execution on the Raspberry Pi Pico W.<br/>"
        "• <b>WebSerial Live Hardware Streaming:</b> Connect physical MyoWare / ADS1115 analog sensors directly to the React dashboard via USB Serial."
    )
    story.append(Paragraph(roadmap_text, body_style))
    story.append(Spacer(1, 10))

    # Signoff footer
    story.append(HRFlowable(width="100%", thickness=1, color=c_border, spaceBefore=4, spaceAfter=8))
    story.append(Paragraph("<b>Report Generated:</b> Real-Time EMG Gesture Recognition & Control Project | Department of IoT Engineering", meta_badge_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated publication-quality PDF report: {filename}")


if __name__ == '__main__':
    build_pdf()
