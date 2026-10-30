"""QR code for the poster footer (pip install 'qrcode[pil]')."""
import qrcode
q = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=20, border=2)
q.add_data("https://github.com/jamesrahenry/ai-day-2026"); q.make(fit=True)
q.make_image(fill_color="#15181D", back_color="white").save("qr_ai_day_2026.png")
