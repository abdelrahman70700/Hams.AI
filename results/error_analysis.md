# Error analysis — `after` configuration

Source: `results/generation_items_after.jsonl` (138 test questions, 26 unanswerable, 45 conversation turns). Every example below is copied verbatim from the benchmark output; categories are assigned by the rules in `scripts/error_analysis.py` (a question can appear in several categories).

| Category | Count |
|---|---|
| Wrong document retrieved | 25 |
| Correct retrieval but unsupported (hallucinated) answer | 11 |
| Missing or incorrect citations | 8 |
| Language drift (answered in the wrong language) | 0 |
| Mixed Arabic/English query handled incorrectly | 1 |
| Conversation memory failure (lost reference / wrong rewrite) | 3 |
| Wrong numbers, dates, prices, names or negation | 8 |
| Untranslated text in the wrong-language output | 1 |
| 'Not found' handled incorrectly (fabricated answer) | 3 |
| False 'not found' (answer was in the retrieved context) | 14 |

## Wrong document retrieved (25)

* **Q** (ar, `02_ar_services:15:0`): ما مواعيد عمل المعارض يوم الجمعة؟
  * **A**: تتوقف إجازات المعارض في آخر أسبوعين من شهر رمضان وخلال أسبوع عروض الجمعة البيضاء، إلا في الحالات الطارئة وبموافقة مدير المنطقة [1].
  * reference: المعارض تعمل من السبت إلى الخميس من 10 صباحاً حتى 11 مساءً، والجمعة من 4 عصراً حتى 11 مساءً، وفي رمضان من 1 ظهراً إلى 5 عصراً ثم من 9 مساءً حتى 2 بعد منتصف الليل.
  * gold: branches_ar#0 · retrieved: hr_leave_policy_ar#1, holiday_calendar_2026_en#0, hr_remote_work_ar#0, sla_support_tiers_en#1 · cited: hr_leave_policy_ar#1
  * note: no gold chunk among the chunks given to the LLM
* **Q** (ar, `02_ar_services:15:2`): المعرض يفتح الجمعة الصبح؟
  * **A**: المعرض يفتح يوم الجمعة من الساعة 9 صباحاً إلى 9 مساءً، ولكن هذا الإجراء ينطبق فقط داخل الرياض، حيث لا تتوفر خدمة التوصيل في ذلك اليوم [2].
  * reference: المعارض تعمل من السبت إلى الخميس من 10 صباحاً حتى 11 مساءً، والجمعة من 4 عصراً حتى 11 مساءً، وفي رمضان من 1 ظهراً إلى 5 عصراً ثم من 9 مساءً حتى 2 بعد منتصف الليل.
  * gold: branches_ar#0 · retrieved: holiday_calendar_2026_en#1, policy_delivery_ar#1, holiday_calendar_2026_en#0, hr_remote_work_ar#0 · cited: policy_delivery_ar#1
  * note: no gold chunk among the chunks given to the LLM
* **Q** (cross, `03_ar_products_hr:21:1`): Do you offer professional sofa cleaning?
  * **A**: Sorry, this information was not found in the knowledge bank.
  * reference: Professional cleaning costs SAR 250 for a three-seater sofa in Riyadh, Jeddah and Dammam.
  * gold: care_guide_ar#1 · retrieved: — · cited: —
  * note: no gold chunk among the chunks given to the LLM

## Correct retrieval but unsupported (hallucinated) answer (11)

* **Q** (cross, `02_ar_services:13:2`): Where is the Malqa branch located?
  * **A**: The Malqa branch is located at Al-Ans bin Malik Street, Al-Malqa neighborhood, Riyadh, Saudi Arabia [1]. It opened on September 15, 2025, and features a 6,000 square meter area dedicated to office furniture [1]. The branch operates from Saturday to Thursday from 10:00 AM to 11:00 PM, with extended hours on Friday from 4:00 PM to 11:00 PM [1].
  * reference: The Al-Malqa branch is on Anas Bin Malik Road in Al-Malqa, Riyadh; it opened on 15 September 2025 and is the largest branch at 6,000 m² with an office furniture section.
  * gold: branches_ar#0 · retrieved: branches_ar#0, shipping_gcc_en#1, incident_severity_guide_en#0, b2b_contract_terms_en#2 · cited: branches_ar#0
  * note: numeric faithfulness 0.8571428571428571, lexical support None
* **Q** (cross, `03_ar_products_hr:2:1`): Do you have a kids bedroom set and what colours does it come in?
  * **A**: Yes, the Saraya Living catalog offers a kids' bedroom set called "Sama" in 2026. It is available in the following colors: light blue, pink, and gray. The price is 4,200 SAR inclusive of tax, though this price does not include the bed frame. [1]
  * reference: The Sama kids' bedroom costs SAR 4,200 and includes a 90×190 single bed with rounded edges, a study desk and a two-door wardrobe, in light blue, pink or grey.
  * gold: product_bedrooms_ar#1 · retrieved: product_bedrooms_ar#1, product_bedrooms_ar#0, smartrest_troubleshooting_mx#1, smartrest_troubleshooting_mx#0 · cited: product_bedrooms_ar#1
  * note: numeric faithfulness 0.5, lexical support None
* **Q** (cross, `03_ar_products_hr:16:2`): Can warehouse staff work from home?
  * **A**: According to the provided sources, warehouse staff are explicitly excluded from the policy allowing support department employees to work from home. The policy states that support staff can work from home for up to two days a week after completing a 90-day probation period, but it specifically notes that this policy does not apply to "warehouse and storage staff" [1]. Additionally, the company's of
  * reference: Support-function staff may work remotely up to two days a week after the 90-day probation, on Sundays or Thursdays only; it does not apply to showroom, warehouse or installation staff.
  * gold: hr_remote_work_ar#0 · retrieved: hr_remote_work_ar#0, hr_leave_policy_ar#0, holiday_calendar_2026_en#0, office_relocation_service_en#0 · cited: hr_remote_work_ar#0, office_relocation_service_en#0
  * note: numeric faithfulness 1.0, lexical support 0.10256410256410256

## Missing or incorrect citations (8)

* **Q** (cross, `06_mixed_rest:36:1`): ماذا يحدث للأثاث القديم الذي تجمعه سرايا؟
  * **A**: تقوم سرايا بتخلص من الأثاث القديم برسوم تتراوح بين 150 ريال للقطعة الكبيرة و 200 ريال للغرفة، بالإضافة إلى رسوم إضافية حسب نوع الخدمة المطلوبة مثل تركيب المطابخ أو تثبيت القطع على الجدران [1].
  * reference: جمع برنامج استرجاع الأثاث 4,300 قطعة في 2025، أعيد تجديد 71% منها والتبرع بها أو بيعها في قسم التصفية، والباقي أعيد تدويره.
  * gold: sustainability_report_2025_en#0 · retrieved: policy_installation_ar#0, b2b_contract_terms_en#2, faq_saraya_business_ar#0, sustainability_report_2025_en#0 · cited: policy_installation_ar#0
  * note: cited chunk is not the gold chunk
* **Q** (cross, `07_test_docs_extra:3:1`): Is there a Saraya showroom near Kingdom Tower in Riyadh?
  * **A**: There is no information in the provided sources about a Saraya showroom near Kingdom Tower in Riyadh.
  * reference: The Olaya branch in Riyadh is on King Fahd Road in Al-Olaya, opposite Kingdom Tower; phone 0114650000.
  * gold: branches_ar#0 · retrieved: faq_saraya_business_ar#0, branches_ar#0, kitchens_catalog_mx#0, holiday_calendar_2026_en#1 · cited: faq_saraya_business_ar#0
  * note: no [n] markers in the answer
* **Q** (cross, `07_test_docs_extra:3:2`): What is the phone number of the Olaya branch?
  * **A**: The phone number for the Olaya branch is not found in the provided sources.
  * reference: The Olaya branch in Riyadh is on King Fahd Road in Al-Olaya, opposite Kingdom Tower; phone 0114650000.
  * gold: branches_ar#0 · retrieved: branches_ar#1, branches_ar#0, enterprise_package_pricing_en#3, faq_online_orders_en#0 · cited: branches_ar#1
  * note: no [n] markers in the answer

## Language drift (answered in the wrong language) (0)

_No cases found in this run._

## Mixed Arabic/English query handled incorrectly (1)

* **Q** (mixed, `07_test_docs_extra:19:1`): طلبي بـ800 ريال، هل ينفع معه كود الترحيب؟
  * **A**: نعم، يمكن استخدام كود الترحيب WELCOME10 لطلب بقيمة 800 ريال، حيث يمنح خصما بنسبة 10% على أول طلب عبر التطبيق، بشرط ألا تقل قيمة الطلب عن 1,000 ريال. [1]
  * reference: الحد الأقصى لخصم كود WELCOME10 هو 300 ريال، والحد الأدنى للطلب 1,000 ريال.
  * gold: pricing_promotions_faq_mx#0 · retrieved: pricing_promotions_faq_mx#0, faq_payment_ar#1, branches_ar#1, delivery_tracking_mx#1 · cited: pricing_promotions_faq_mx#0

## Conversation memory failure (lost reference / wrong rewrite) (3)

* **Q** (interactive, `conv02:3`): طيب وإذا ما التزموا بالوقت، وش التعويض؟
  * rewritten: إذا ما التزموا بالوقت، وش التعويض؟  (reference: ما التعويض إذا لم تلتزم سرايا بوقت الـSLA لتذاكر الدعم؟)
  * **A**: إذا تم الالتزام بالوقت، يحصل العميل على تعويض تلقائي دون الحاجة لتقديم طلب، حيث تتراوح المدة بين 0 إلى 3 أيام عمل مع قيمة تعويض تتراوح بين 50 إلى 200 ريال [1].
  * gold: sla_support_tiers_en#2 · retrieved: delivery_late_compensation_ar#0, crm_guide_agents_mx#0, delivery_late_compensation_ar#1, expense_policy_en#2 · cited: delivery_late_compensation_ar#0
* **Q** (interactive, `conv10:3`): وكم ضمانها؟
  * rewritten: كم ضمانها للإrgoPro Plus (EP-300)؟  (reference: كم مدة الضمان على كراسي ErgoPro المكتبية؟)
  * **A**: يتميز ErgoPro Plus (EP-300) بضمان لمدة 3 سنوات على محرك الرفع [1].
  * gold: policy_warranty_ar#0 · retrieved: product_office_chairs_en#0, smartrest_troubleshooting_mx#0, smartrest_troubleshooting_mx#2, incident_severity_guide_en#1 · cited: product_office_chairs_en#0
* **Q** (interactive, `conv13:2`): وإذا ما عجبتني كم يخصمون؟
  * rewritten: إذا ما عجبتني كم يخصمون؟  (reference: كم يخصم من المبلغ عند إرجاع المرتبة ضمن برنامج التجربة؟)
  * **A**: يتم خصم المبلغ فوراً عند تأكيد الطلب للدفع الإلكتروني، بينما يتم دفع 50% عند الطلب و50% قبل التوصيل للمنتجات المصنوعة حسب الطلب [2].
  * gold: mattress_trial_ar#0 · retrieved: pricing_promotions_faq_mx#0, faq_payment_ar#0, pricing_promotions_faq_mx#1, privacy_data_retention_en#2 · cited: faq_payment_ar#0

## Wrong numbers, dates, prices, names or negation (8)

* **Q** (cross, `02_ar_services:13:2`): Where is the Malqa branch located?
  * **A**: The Malqa branch is located at Al-Ans bin Malik Street, Al-Malqa neighborhood, Riyadh, Saudi Arabia [1]. It opened on September 15, 2025, and features a 6,000 square meter area dedicated to office furniture [1]. The branch operates from Saturday to Thursday from 10:00 AM to 11:00 PM, with extended hours on Friday from 4:00 PM to 11:00 PM [1].
  * reference: The Al-Malqa branch is on Anas Bin Malik Road in Al-Malqa, Riyadh; it opened on 15 September 2025 and is the largest branch at 6,000 m² with an office furniture section.
  * gold: branches_ar#0 · retrieved: branches_ar#0, shipping_gcc_en#1, incident_severity_guide_en#0, b2b_contract_terms_en#2 · cited: branches_ar#0
  * note: numbers not in the sources: ['00']
* **Q** (cross, `03_ar_products_hr:2:1`): Do you have a kids bedroom set and what colours does it come in?
  * **A**: Yes, the Saraya Living catalog offers a kids' bedroom set called "Sama" in 2026. It is available in the following colors: light blue, pink, and gray. The price is 4,200 SAR inclusive of tax, though this price does not include the bed frame. [1]
  * reference: The Sama kids' bedroom costs SAR 4,200 and includes a 90×190 single bed with rounded edges, a study desk and a two-door wardrobe, in light blue, pink or grey.
  * gold: product_bedrooms_ar#1 · retrieved: product_bedrooms_ar#1, product_bedrooms_ar#0, smartrest_troubleshooting_mx#1, smartrest_troubleshooting_mx#0 · cited: product_bedrooms_ar#1
  * note: numbers not in the sources: ['2026']
* **Q** (cross, `05_en_internal_products:13:1`): هل يجب تثبيت الدولاب في الجدار؟
  * **A**: يجب تثبيت الدولاب في الجدار إذا كان عرض الدولاب أكثر من 3 أمتار، حيث يتم تحديد رسوم إضافية لهذه الخدمة بـ 120 ريال لكل قطعة [1].
  * reference: تثبيت دعامة منع الانقلاب (الكيس F) على الجدار خطوة إلزامية للسلامة عند تركيب دولاب لمار LW-3D.
  * gold: assembly_manual_wardrobe_en#1 · retrieved: policy_installation_ar#0, faq_saraya_business_ar#1, policy_installation_ar#1, product_bedrooms_ar#1 · cited: policy_installation_ar#0
  * note: numbers not in the sources: ['3']

## Untranslated text in the wrong-language output (1)

* **Q** (interactive, `conv06:3`): وكم البدل اليومي؟
  * rewritten: كم البدل اليومي؟  (reference: كم البدل اليومي للسفر الداخلي للموظفين؟)
  * **A**: يحق لموظفي الإدارات المساندة العمل عن بعد يومين في الأسبوع كحد أقصى بعد إكمال فترة التجربة البالغة 90 يوماً [1].
يتم صرف بدل إنترنت شهري قدره 150 ريالا للموظفين المعتمدين للعمل عن بعد [1].
بعض العروض حصرية Online-only وتظهر في التطبيق فقط [3].
Saraya Business provides furnished workspaces as a subscription: furniture, installation, maintenance, and replacements for a fixed monthly fee [4].
  * gold: expense_policy_en#2 · retrieved: hr_remote_work_ar#0, expense_policy_en#2, pricing_promotions_faq_mx#1, enterprise_package_pricing_en#0 · cited: hr_remote_work_ar#0, pricing_promotions_faq_mx#1, enterprise_package_pricing_en#0
  * note: 14 Latin-script words not present in the question: ['Online', 'only', 'Saraya', 'Business', 'provides', 'furnished']

## 'Not found' handled incorrectly (fabricated answer) (3)

* **Q** (unanswerable, `unans:14`): Does the ErgoPro Lite chair come in white?
  * **A**: The ErgoPro Lite chair is not available in white, as the product sheet only lists black and grey colors [1].
  * gold: — · retrieved: product_office_chairs_en#0, promotions_white_friday_2026_mx#1, smartrest_troubleshooting_mx#2, smartrest_troubleshooting_mx#0 · cited: product_office_chairs_en#0
* **Q** (unanswerable, `unans:26`): كم وزن كرسي ErgoPro Plus؟
  * **A**: يبلغ وزن كرسي ErgoPro Plus 130 كجم [1].
  * gold: — · retrieved: product_office_chairs_en#0, smartrest_troubleshooting_mx#0, smartrest_troubleshooting_mx#2, enterprise_package_pricing_en#3 · cited: product_office_chairs_en#0
* **Q** (unanswerable, `unans:29`): ما هي عروض اليوم الوطني 2026؟
  * **A**: تاريخ اليوم الوطني 2026 هو يوم الأحد 23 سبتمبر، ويبدأ العمل فيه لمدة يوم واحد [1].
  * gold: — · retrieved: holiday_calendar_2026_en#0, ramadan_offers_2026_ar#0, promotions_white_friday_2026_mx#0, enterprise_package_pricing_en#3 · cited: holiday_calendar_2026_en#0

## False 'not found' (answer was in the retrieved context) (14)

* **Q** (cross, `02_ar_services:15:1`): What are your showroom opening hours during Ramadan?
  * **A**: Sorry, this information was not found in the knowledge bank.
  * reference: Showrooms open Saturday–Thursday 10 AM–11 PM and Friday 4 PM–11 PM; in Ramadan 1–5 PM and 9 PM–2 AM.
  * gold: branches_ar#0 · retrieved: holiday_calendar_2026_en#1, branches_ar#0, ramadan_offers_2026_ar#0, holiday_calendar_2026_en#0 · cited: —
* **Q** (cross, `03_ar_products_hr:3:1`): Which wardrobe model comes with the Lamar bedroom?
  * **A**: Sorry, this information was not found in the knowledge bank.
  * reference: The Lamar classic bedroom costs SAR 9,450 and includes an upholstered-headboard bed, the LW-3D 3-door wardrobe, a dresser, two nightstands and a storage bench.
  * gold: product_bedrooms_ar#0 · retrieved: assembly_manual_wardrobe_en#0, assembly_manual_wardrobe_en#1, product_bedrooms_ar#0, product_bedrooms_ar#1 · cited: —
* **Q** (cross, `05_en_internal_products:2:1`): نسيت أرفع مصاريف رحلة عمل من شهرين، هل يتم تعويضي؟
  * **A**: عذراً، لم أجد هذه المعلومة في قاعدة المعرفة.
  * reference: تقدم مطالبات المصاريف مع الإيصالات خلال 30 يوماً من العودة، ولا تعوض المطالبات بعد 60 يوماً، وتصرف مع الراتب إذا اعتمدت قبل يوم 20 من الشهر.
  * gold: expense_policy_en#2 · retrieved: expense_policy_en#2, sla_support_tiers_en#2, delivery_late_compensation_ar#0, b2b_contract_terms_en#1 · cited: —
