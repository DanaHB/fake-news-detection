from locust import HttpUser, task, between

class FakeNewsAppUser(HttpUser):
    # وقت الانتظار بين كل ضغطة وضغطة للمستخدم (بين 1 إلى 3 ثوانٍ)
    wait_time = between(1, 3)

    @task
    def check_news_page(self):
        # 1. محاكاة الدخول على الصفحة الرئيسية للموقع
        self.client.get("/") 

    @task
    def simulate_prediction(self):
        # 2. محاكاة إرسال نص فحص (إذا كان كود ستريمليت يستقبل بيانات)
        # ملاحظة: يمكنك الاكتفاء بـ get("/") لفحص قدرة السيرفر على فتح الموقع لعدد كبير
        self.client.get("/")