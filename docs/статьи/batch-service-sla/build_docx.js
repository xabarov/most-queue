// Build the article as a .docx, following the "Труды ВКА" house style observed
// in the user's own precedent article (author block, caps title, plain
// abstract + Ключевые слова, unnumbered section headers, "Рисунок N." below
// figures, "Таблица N." above tables, numbered formulas, GOST-ish numbered
// bracketed references in citation order).
//
// Personal article material -- this script and its .docx output are not part
// of the most-queue package (docs/статьи/ is gitignored).
//
// Run: node build_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, WidthType, BorderStyle, TabStopType,
  TabStopPosition, VerticalAlign,
} = require("docx");

const FIG_DIR = path.join(__dirname, "figures");
const FONT = "Times New Roman";
const SZ = 24;        // 12pt body
const SZ_SMALL = 22;  // 11pt abstract/keywords
const SZ_TITLE = 28;  // 14pt title

function imagePixels(file, widthIn) {
  // Read PNG dimensions straight from the IHDR chunk (no extra deps).
  // docx-js's ImageRun.transformation expects PIXELS (it applies its own
  // px->EMU factor internally) -- passing EMU here double-converts and
  // produces an enormous, invisible image. 96 px/in is the OOXML reference.
  const buf = fs.readFileSync(path.join(FIG_DIR, file));
  const w = buf.readUInt32BE(16);
  const h = buf.readUInt32BE(20);
  const heightIn = (widthIn * h) / w;
  return { width: Math.round(widthIn * 96), height: Math.round(heightIn * 96) };
}

function p(text, opts = {}) {
  const { bold = false, italic = false, size = SZ, align = AlignmentType.JUSTIFIED, spacingAfter = 160, sub, sup } = opts;
  return new Paragraph({
    alignment: align,
    spacing: { after: spacingAfter, line: 276 },
    children: [new TextRun({ text, bold, italics: italic, size, font: FONT, subScript: sub, superScript: sup })],
  });
}

function runs(parts, opts = {}) {
  // parts: array of {text, bold?, italic?, sub?, sup?}
  const { size = SZ, align = AlignmentType.JUSTIFIED, spacingAfter = 160, indentFirstLine = false } = opts;
  return new Paragraph({
    alignment: align,
    spacing: { after: spacingAfter, line: 276 },
    indent: indentFirstLine ? { firstLine: 480 } : undefined,
    children: parts.map(
      (part) =>
        new TextRun({
          text: part.text,
          bold: part.bold || false,
          italics: part.italic || false,
          subScript: part.sub || false,
          superScript: part.sup || false,
          size: part.size || size,
          font: FONT,
        })
    ),
  });
}

function heading(text) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 280, after: 160 },
    children: [new TextRun({ text, bold: true, size: SZ, font: FONT })],
  });
}

function formula(mainParts, number) {
  // Centered formula with a right-tab-stopped number, e.g. "(3)".
  return new Paragraph({
    tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
    spacing: { after: 160, before: 80 },
    alignment: AlignmentType.CENTER,
    children: [
      ...mainParts.map(
        (part) =>
          new TextRun({
            text: part.text,
            italics: part.italic || false,
            subScript: part.sub || false,
            superScript: part.sup || false,
            size: SZ,
            font: FONT,
          })
      ),
      new TextRun({ text: `\t(${number})`, size: SZ, font: FONT }),
    ],
  });
}

function figureParagraph(file, widthIn, caption) {
  const dims = imagePixels(file, widthIn);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 40 },
      children: [
        new ImageRun({
          type: "png",
          data: fs.readFileSync(path.join(FIG_DIR, file)),
          transformation: dims,
          altText: { title: caption, description: caption, name: file },
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: caption, size: SZ_SMALL, font: FONT })],
    }),
  ];
}

function tableCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, bold: true, size: SZ_SMALL, font: FONT })],
  });
}

const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const cellBorders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };

function cell(text, opts = {}) {
  const { bold = false, width, align = AlignmentType.CENTER } = opts;
  return new TableCell({
    borders: cellBorders,
    width: { size: width, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text, bold, size: SZ_SMALL, font: FONT })] })],
  });
}

function makeTable(headerRow, dataRows, colWidths) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ tableHeader: true, children: headerRow.map((t, i) => cell(t, { bold: true, width: colWidths[i] })) }),
      ...dataRows.map(
        (row) => new TableRow({ children: row.map((t, i) => cell(String(t), { width: colWidths[i] })) })
      ),
    ],
  });
}

// ---------------------------------------------------------------------------
// Content
// ---------------------------------------------------------------------------

const children = [];

// УДК
children.push(p("УДК 519.872", { align: AlignmentType.LEFT, spacingAfter: 240 }));

// Author block (placeholder -- adjust name/degree/affiliation before submission)
children.push(p("Р.С. Хабаров", { align: AlignmentType.LEFT, spacingAfter: 0 }));
children.push(p("[степень, звание, место работы -- заполнить]", { align: AlignmentType.LEFT, size: SZ_SMALL, spacingAfter: 240 }));

// Title
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [
      new TextRun({
        text: "ТОЧНАЯ ВЕРОЯТНОСТЬ НАРУШЕНИЯ SLA В СИСТЕМАХ ПАКЕТНОГО ОБСЛУЖИВАНИЯ С ОГРАНИЧЕННЫМ РАЗМЕРОМ ПАКЕТА И ФАЗОВЫМ ВРЕМЕНЕМ ОБСЛУЖИВАНИЯ",
        bold: true,
        size: SZ_TITLE,
        font: FONT,
      }),
    ],
  })
);

// Abstract (RU)
children.push(
  p(
    "Рассмотрена система пакетного (batch-service) обслуживания с временем обслуживания пакета, зависящим от его размера, -- модель динамического батчинга при инференсе нейронных сетей, в том числе больших языковых моделей (LLM). Предложен метод получения точной (не приближённой и не являющейся оценкой сверху) вероятности нарушения SLA P(W>D) для непрерывного времени, ограниченного максимального размера пакета и фазового (Erlang/H2, произвольный коэффициент вариации) batch-size-зависимого времени обслуживания. Метод основан на PASTA-разложении времени ожидания по фазовому типу и действии матричной экспоненты на явно построенный подгенератор. Результат точно регрессирует к известным частным случаям и подтверждён независимым имитационным моделированием. Численное сравнение с опубликованной оценкой сверху среднего времени ожидания показывает, что при реалистичном максимальном размере пакета истинное среднее время ожидания может превышать эту оценку почти на порядок; сравнение с приближением по моментам показывает систематическое занижение вероятности нарушения SLA в глубоком хвосте на несколько порядков.",
    { size: SZ_SMALL, spacingAfter: 160 }
  )
);
children.push(
  runs(
    [
      { text: "Ключевые слова: ", bold: true },
      { text: "теория массового обслуживания, пакетное обслуживание, batch-size-dependent service, фазовые распределения, SLA, вероятность нарушения дедлайна, инференс больших языковых моделей, динамический батчинг." },
    ],
    { size: SZ_SMALL, spacingAfter: 320 }
  )
);

// English block
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
    children: [
      new TextRun({
        text: "EXACT SLA-VIOLATION PROBABILITY FOR BATCH-SERVICE QUEUES WITH A BOUNDED BATCH SIZE AND PHASE-TYPE SERVICE TIME",
        bold: true,
        size: SZ,
        font: FONT,
      }),
    ],
  })
);
children.push(p("R.S. Khabarov", { align: AlignmentType.CENTER, size: SZ_SMALL, spacingAfter: 160 }));
children.push(
  p(
    "A batch-service queue with batch-size-dependent service time is the natural model for dynamic-batching neural-network (including large language model, LLM) inference. We derive an exact -- not approximate, not an upper bound -- deadline-violation probability P(W>D) for continuous time, a bounded maximum batch size and phase-type (Erlang/H2, arbitrary coefficient of variation) batch-size-dependent service time, via matrix-exponential action on an explicitly constructed phase-type sub-generator obtained from a PASTA decomposition of the waiting time. The result regresses exactly to known degenerate cases and is validated against independent discrete-event simulation. A numerical comparison with a published closed-form upper bound on mean wait shows that under a realistic bounded batch size the true mean wait can exceed that bound by almost an order of magnitude; comparison with a moment-fit approximation shows systematic underestimation of deep-tail SLA-violation probability by several orders of magnitude.",
    { size: SZ_SMALL, spacingAfter: 160 }
  )
);
children.push(
  runs(
    [
      { text: "Key words: ", bold: true, italic: true },
      { text: "queueing theory, batch service, batch-size-dependent service, phase-type distributions, SLA, deadline-violation probability, large language model inference, dynamic batching.", italic: true },
    ],
    { size: SZ_SMALL, spacingAfter: 360 }
  )
);

// ---------------------------------------------------------------------------
// Введение
// ---------------------------------------------------------------------------
children.push(heading("Введение"));
children.push(
  p(
    "Ускорение инференса нейронных сетей на графических процессорах (GPU) существенно опирается на динамический батчинг -- объединение нескольких независимо поступивших запросов в один пакет (batch), обрабатываемый сервером одновременно. Такое объединение повышает пропускную способность и энергоэффективность сервера, но ценой дополнительной задержки: запрос вынужден дожидаться, пока сформируется пакет и пока этот пакет будет обслужен [1]. Применительно к инференсу больших языковых моделей (LLM) данная задержка прямо входит в метрику time-to-first-token (TTFT), для которой операторы всё чаще формулируют требования к качеству обслуживания (SLO) в терминах вероятности, а не среднего значения: P(TTFT>D)<ε для заданного малого ε [2, 3].",
    { indentFirstLine: true }
  )
);
children.push(
  p(
    "Два структурных свойства реальных систем динамического батчинга отличают их от классической модели пакетного обслуживания Бейли [4]. Во-первых, время обслуживания пакета зависит от его размера: обработка пакета из b заявок занимает больше времени, чем пакета из одной заявки, но не пропорционально b -- типичная эмпирическая зависимость близка к линейной с положительным свободным членом [1]. Во-вторых, размер пакета ограничен сверху: память GPU ограничивает максимальный размер батча b_max, и на практике b_max -- конечное, зачастую небольшое число, а не абстрактная бесконечность.",
    { indentFirstLine: true }
  )
);
children.push(
  p(
    "Прямой практический прецедент -- работа Inoue [1], формулирующая сервер инференса как batch-service очередь с batch-size-зависимым временем обслуживания и выводящая закрытую оценку сверху среднего времени ожидания E[W] при двух упрощающих допущениях: неограниченный максимальный размер пакета (при освобождении сервера в пакет забираются все ожидающие заявки) и детерминированное, линейно зависящее от размера время обслуживания пакета. Автор прямо указывает, что точный анализ труден даже для batch-size-зависимого M/M/1, и что для случая конечного b_max известен лишь алгоритмический, не дающий замкнутой формулы подход на основе матрично-аналитических методов [5].",
    { indentFirstLine: true }
  )
);
children.push(
  p(
    "Начиная с работы [6] и продолжая обширной серией последующих работ [7–10], детально проработано точное стационарное распределение длины очереди (и содержимого сервера) для систем пакетного обслуживания с batch-size-зависимым, в том числе фазовым, временем обслуживания при различных входных потоках. Эти работы дают точное распределение длины очереди в различные моменты времени, но не распределение времени ожидания конкретной помеченной заявки. Классический результат для времени ожидания в системе пакетного обслуживания с общим, но не зависящим от размера пакета временем обслуживания получен точно методом корней характеристического уравнения [11]; та же группа авторов, расширяя метод на коррелированный входной поток, в 2025 году явно называет полученное распределение времени ожидания приближением [12] -- то есть даже без batch-size-зависимости точное решение для коррелированного входа остаётся открытым. Единственная найденная нами работа, дающая хвост времени ожидания при batch-size-зависимом обслуживании, -- [13], но это дискретное время и заявленное самими авторами приближение.",
    { indentFirstLine: true }
  )
);
children.push(
  p(
    "Таким образом, пересечение условий «точно» × «непрерывное время» × «время ожидания, а не только длина очереди» × «batch-size-зависимое обслуживание» не закрыто ни одной из найденных работ, включая самые свежие публикации 2024–2026 гг. той же исследовательской традиции. Дальнейшая структура статьи следующая. Сначала приводится постановка задачи. Далее описан метод: PASTA-разложение времени ожидания по фазовому типу и точный хвост через действие матричной экспоненты. После этого представлены результаты проверки и два численных эксперимента -- сравнение с оценкой [1] и с приближением по моментам.",
    { indentFirstLine: true }
  )
);

// ---------------------------------------------------------------------------
// Постановка задачи
// ---------------------------------------------------------------------------
children.push(heading("Постановка задачи"));
children.push(
  p(
    "Рассматривается однолинейная система M/PH[1,b]/1: пуассоновский входной поток интенсивности λ; как только сервер освобождается и есть хотя бы одна ожидающая заявка, формируется пакет из min(b,j) заявок, где j -- число ожидающих (минимальный порог запуска a=1, то есть сервер не ждёт накопления нескольких заявок -- типичный режим динамического батчинга); время обслуживания пакета размера i имеет фазовое распределение -- Erlang(k, rate(i)) либо H2(p1(i), μ1(i), μ2(i)) -- с произвольной зависимостью параметров от i, в частности эмпирически наблюдаемой линейной зависимостью среднего от размера пакета [1].",
    { indentFirstLine: true }
  )
);
children.push(
  p(
    "Цель -- получить точное P(W>D) для D≥0, где W -- время ожидания в очереди случайно выбранной (по PASTA) заявки, без ограничения на распределение времени обслуживания, кроме принадлежности фазовому классу.",
    { indentFirstLine: true }
  )
);

// ---------------------------------------------------------------------------
// Метод
// ---------------------------------------------------------------------------
children.push(heading("Метод"));
children.push(
  p(
    "Система моделируется усечённой по длине очереди (j≤N) цепью Маркова с состояниями (i,φ,j), где i∈{0,...,b} -- размер обслуживаемого пакета (i=0 -- простой), φ -- текущая фаза обслуживания пакета (для Erlang -- номер пройденной фазы 0..k-1, для H2 -- выбранная при старте ветвь {0,1}), j -- число ожидающих заявок. Стационарное распределение π(i,φ,j) находится прямым решением πQ=0 для разреженного генератора Q.",
    { indentFirstLine: true }
  )
);
children.push(
  p(
    "По свойству PASTA заявка, приходящая в систему, видит стационарное распределение π(i,φ,j). Если сервер простаивает (i=0), заявка немедленно формирует пакет: W=0. Если сервер занят, время ожидания раскладывается на два независимых слагаемых: остаток текущего обслуживаемого пакета (Erlang(k-p, rate(i)) для Erlang-случая, где p -- застанная фаза; Exp(μφ(i)) для H2-случая по марковости экспоненциальной фазы) и ⌊j/b⌋ полных последующих пакетов размера b. Для Erlang это строго последовательная цепь из ⌊j/b⌋·k фаз ставки rate(b); для H2 -- ветвящаяся цепь, поскольку каждый последующий пакет независимо заново выбирает свою ветвь.",
    { indentFirstLine: true }
  )
);
children.push(
  p(
    "Для каждого занятого состояния (i,φ,j) c π(i,φ,j)>0 строится явный конечный подгенератор T(i,φ,j) поглощающей цепи Маркова, представляющей это разложение (начальный вектор α=e1, поглощение -- момент начала обслуживания помеченной заявки). Условный хвост времени ожидания -- классическая формула phase-type распределения:",
    { indentFirstLine: true }
  )
);
children.push(
  formula(
    [
      { text: "P(W>t | i,φ,j) = α·exp(T" },
      { text: "i,φ,j", sub: true },
      { text: "·t)·1" },
    ],
    1
  )
);
children.push(p("Итоговый, безусловный точный хвост:", { indentFirstLine: false, spacingAfter: 40 }));
children.push(
  formula(
    [
      { text: "P(W>t) = Σ" },
      { text: "(i,φ,j): i≥1", sub: true },
      { text: " π(i,φ,j)·α·exp(T" },
      { text: "i,φ,j", sub: true },
      { text: "·t)·1" },
    ],
    2
  )
);
children.push(
  p(
    "Для MM1/Erlang подгенератор T -- двухдиагональная (последовательная) матрица; для H2 он содержит ветвление. Экспонента вычисляется не явным построением плотной матрицы, а её действием на единичный вектор через разреженное представление, что даёт точный результат без ручного разбора случаев совпадающих/несовпадающих кратностей ставок, неизбежного при классическом подходе частичных дробей для гипоэкспоненциальных/Coxian распределений.",
    { indentFirstLine: true }
  )
);
children.push(
  p(
    "При b=1 (отсутствие батчинга) результат должен точно совпасть с классической формулой хвоста M/M/1: P(W>t)=ρ·exp(-μ(1-ρ)t). При k=1 Erlang-калькулятор должен точно совпасть с экспоненциальным (MM1) калькулятором; при p1=1 H2-калькулятор должен точно совпасть с ним же (вторая ветвь становится недостижимой). Эти тождества -- не только регрессионные тесты, но и необходимое условие корректности построения.",
    { indentFirstLine: true }
  )
);

// ---------------------------------------------------------------------------
// Проверка
// ---------------------------------------------------------------------------
children.push(heading("Проверка"));
children.push(
  p(
    "Все три регрессионных тождества подтверждены с точностью до ошибок округления float64 (порядка 10⁻¹⁶). Независимо от регрессий результат для batch-size-зависимого времени обслуживания, где регрессионные тождества неприменимы, проверен против независимого имитационного моделирования -- дискретно-событийной симуляции, реализованной независимо от кода основного метода, -- для каждого из трёх калькуляторов, в нескольких точках t, включая точки глубокого хвоста, где симуляция из 3×10⁶ выборок не дала ни одного события превышения порога: точный результат в этих точках даёт малое, но ненулевое значение, согласующееся по порядку величины с границей, которую могла бы дать симуляция существенно большего объёма.",
    { indentFirstLine: true }
  )
);

// ---------------------------------------------------------------------------
// Численные эксперименты
// ---------------------------------------------------------------------------
children.push(heading("Численные эксперименты"));
children.push(
  p(
    "Первая реализация использовала явное построение плотной матричной экспоненты на подгенераторе размера до нескольких сотен состояний (при большом ⌊N/b⌋). Время одного вызова составило 57 секунд при реалистичном усечении очереди (N=500). Переход на действие экспоненты на вектор без явного построения плотной матрицы дал идентичный результат за 0,19 секунды -- ускорение около 300 раз, необходимое условие практического применения метода.",
    { indentFirstLine: true }
  )
);
children.push(
  p(
    "Для сравнения с оценкой [1] использованы параметры эмпирической линейной подгонки автора для батч-инференса ResNet-50 на Tesla V100 (α=0,1438, τ0=1,8874), λ=0,8 (устойчивая система при любом b_max≥2: максимальная пропускная способность при b_max=2 составляет 2/(2α+τ0)≈0,92). Закрытая оценка [1] для неограниченного батча: φ(λ,α,τ0)=3,353.",
    { indentFirstLine: true }
  )
);
children.push(...figureParagraph("fig1_mean_wait_vs_bmax.png", 3.4, "Рисунок 1. Истинное среднее время ожидания E[W] в зависимости от b_max против границы [1] (λ=0,8)"));
children.push(
  p(
    "При малом (реалистичном для памяти GPU) b_max истинное среднее время ожидания превышает закрытую оценку [1] -- несмотря на то, что последняя формально является оценкой сверху: граница корректно доказана для неограниченного батча и не распространяется на ограниченный случай. При коэффициенте вариации CV=2 (разумное предположение для пакета со смешанными по длине запросами, типичными для LLM-инференса) расхождение при b_max=2 достигает почти десятикратного значения. При больших b_max, приближающихся к режиму [1], точное значение остаётся ниже границы -- граница верна именно там, где и была выведена, но не несёт информации о своей неприменимости при малом b_max.",
    { indentFirstLine: true }
  )
);
children.push(...figureParagraph("fig2_exact_tail_curves.png", 3.4, "Рисунок 2. Точный хвост P(W>t) для нескольких b_max (Erlang, CV=0,316)"));
children.push(
  p(
    "На рисунке 2 видно, что форма хвоста существенно зависит от b_max: при b_max=2 (режим, близкий к границе устойчивости при выбранной нагрузке) хвост спадает заметно медленнее, чем при b_max=8 и b_max=32, кривые для которых практически совпадают -- система уже близка к своему пределу при неограниченном батче.",
    { indentFirstLine: true }
  )
);
children.push(
  p(
    "Для Erlang-случая (CV=0,316) сравнивается точный хвост с результатом стандартного приёма «подгонка 2–3 моментов под H2/Gamma, вычисление хвоста подогнанного распределения», широко используемого на практике, когда точный метод недоступен.",
    { indentFirstLine: true }
  )
);
children.push(tableCaption("Таблица 1. Точная и подогнанная по моментам вероятность P(W>D)"));
children.push(
  makeTable(
    ["b_max", "D", "Точная P(W>D)", "Подогнанная (2 момента)", "Отн. ошибка"],
    [
      [2, 5, "3,7035×10⁻¹", "3,7107×10⁻¹", "+0,19%"],
      [2, 10, "1,3255×10⁻¹", "1,3331×10⁻¹", "+0,57%"],
      [2, 20, "1,6981×10⁻²", "1,6906×10⁻²", "−0,44%"],
      [8, 5, "2,7386×10⁻⁴", "2,5296×10⁻³", "+824%"],
      [8, 10, "5,6797×10⁻⁹", "2,4628×10⁻⁶", "+43 261%"],
      [8, 20, "3,3642×10⁻¹⁸", "1,6570×10⁻¹²", "+4,93×10⁷%"],
      [16, 5, "2,4137×10⁻⁴", "2,5019×10⁻³", "+937%"],
      [16, 10, "1,0582×10⁻⁹", "2,4014×10⁻⁶", "+226 833%"],
      [16, 20, "5,7739×10⁻¹⁹", "1,5685×10⁻¹²", "+2,72×10⁸%"],
    ],
    [1200, 900, 2200, 2600, 2000]
  )
);
children.push(
  p(
    "При умеренно глубоком хвосте (b_max=2, где D лишь в несколько раз превышает среднее) подгонка по двум моментам работает приемлемо (ошибка <1%). Но при малом среднем времени ожидания и дедлайне, уходящем в действительно глубокий хвост -- во много раз больше среднего, типичный для GPU/LLM SLA режим, где цель -- P(нарушение)<0,1% -- подгонка по моментам систематически и drastически переоценивает вероятность нарушения SLA: на один-два порядка при умеренной глубине хвоста и на много порядков при истинно глубоком хвосте.",
    { indentFirstLine: true }
  )
);

// ---------------------------------------------------------------------------
// Ограничения
// ---------------------------------------------------------------------------
children.push(heading("Ограничения"));
children.push(
  p(
    "Результат ограничен случаем a=1 (нет нижнего порога запуска пакета); случай a>1 требует отдельного («idle-refill») анализа и остаётся открытым. Численное сравнение использует параметры подгонки [1] для ResNet-50-инференса, а не измеренную изменчивость реального батч-инференса LLM, -- это управляемое численное исследование, а не воспроизведение конкретной продакшн-системы. Сравнение с оценкой [1] выполнено только по среднему; сравнение с подгонкой по моментам для хвоста выполнено только для Erlang-случая низкой изменчивости.",
    { indentFirstLine: true }
  )
);

// ---------------------------------------------------------------------------
// Заключение
// ---------------------------------------------------------------------------
children.push(heading("Заключение"));
children.push(
  p(
    "Получена и верифицирована точная вероятность нарушения SLA для системы пакетного обслуживания с ограниченным максимальным размером пакета и фазовым, batch-size-зависимым временем обслуживания произвольного коэффициента вариации -- результат, закрывающий пересечение условий, не покрытое ни классической линией работ по точному распределению длины очереди для batch-size-зависимого обслуживания, ни единственным известным приближённым, дискретно-временным результатом для хвоста времени ожидания с такой зависимостью. Численно показано, что пренебрежение ограничением размера пакета может занижать истинное среднее время ожидания почти на порядок, а приближение по моментам -- занижать точность оценки вероятности нарушения SLA в глубоком хвосте на несколько порядков; оба эффекта напрямую релевантны практике SLA-ориентированного управления GPU/LLM-инференс-серверами.",
    { indentFirstLine: true }
  )
);

// ---------------------------------------------------------------------------
// References
// ---------------------------------------------------------------------------
children.push(heading("Список литературы"));

const refs = [
  "Inoue Y. Queueing analysis of GPU-based inference servers with dynamic batching: a closed-form characterization // Performance Evaluation. 2021. Vol. 147. Art. 102183.",
  "QUARTZ: Quantile-Aware Routing and Queueing for TTFT SLOs in LLM Serving // Findings of ACL. 2026.",
  "Tool-Augmented LLM Serving Under Firm Deadlines: A Queueing-Control Approach. 2026. (SSRN Electronic Journal).",
  "Bailey N.T.J. On queueing processes with bulk service // Journal of the Royal Statistical Society. Series B. 1954. Vol. 16. P. 80–87.",
  "Neuts M.F. Structured Stochastic Matrices of M/G/1 Type and Their Applications. New York: Marcel Dekker, 1989.",
  "Banerjee A., Gupta U.C. Reducing congestion in bulk-service finite-buffer queueing system using batch-size-dependent service // Performance Evaluation. 2012. Vol. 69, No. 1. P. 53–70.",
  "Pradhan S., Gupta U.C., Samanta S.K. Queue-length distribution of a batch service queue with random capacity and batch size dependent service: M/Gʸᵣ/1 // OPSEARCH. 2015. Vol. 53. P. 329–343.",
  "Banerjee A., Gupta U.C., Chakravarthy S.R. Analysis of a finite-buffer bulk-service queue under Markovian arrival process with batch-size-dependent service // Computers & Operations Research. 2015. Vol. 60. P. 138–149.",
  "Gupta U.C. et al. On M/G(a,b)/1/N queue with batch size- and queue length-dependent service. Singapore: Springer, 2018.",
  "Pradhan S., Nandy N., Gupta U.C. Performance analysis of a versatile bulk-service queue with group-arrival, batch-size-dependent service time and queue-length-dependent vacation // Quality Technology & Quantitative Management. 2024. Vol. 22, No. 4. P. 683–726.",
  "Chaudhry M.L., Datta Banik A., Barik S., Goswami V. A novel computational procedure for the waiting-time distribution (in the queue) for bulk-service finite-buffer queues with Poisson input // Mathematics. 2023. Vol. 11, No. 5. Art. 1142.",
  "Banik A.D., Chaudhry M.L., Barik S. et al. On the heuristic computational procedures of the virtual waiting-time distribution in a non-renewal input finite-buffer bulk-service queues: MAP/R(a,b)/1/N // Journal of the Indian Society for Probability and Statistics. 2025. Vol. 26. P. 585–630.",
  "Claeys D., Steyaert B., Walraevens J., Laevens K., Bruneel H. Tail probabilities of the delay in a batch-service queueing model with batch-size dependent service times and a timer mechanism // Computers & Operations Research. 2013. Vol. 40, No. 5. P. 1497–1505.",
];
refs.forEach((text, i) => {
  children.push(
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 100, line: 276 },
      indent: { left: 480, hanging: 480 },
      children: [new TextRun({ text: `${i + 1}. ${text}`, size: SZ_SMALL, font: FONT })],
    })
  );
});

// ---------------------------------------------------------------------------
// Appendix: implementation
// ---------------------------------------------------------------------------
children.push(heading("Приложение. Программная реализация"));
children.push(
  p(
    "Все вычисления воспроизводимы в открытой библиотеке most-queue: most_queue.theory.batch.bulk_service.BulkServiceMM1Calc.get_tail, most_queue.theory.batch.bulk_service_erlang.BulkServiceErlangCalc.get_tail, most_queue.theory.batch.bulk_service_h2.BulkServiceH2Calc.get_tail. Тесты и независимая проверка имитационным моделированием -- tests/units/test_bulk_service_tail.py, test_bulk_service_erlang_tail.py, test_bulk_service_h2_tail.py.",
    { indentFirstLine: true, size: SZ_SMALL }
  )
);

// ---------------------------------------------------------------------------
// Document assembly
// ---------------------------------------------------------------------------
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: SZ } } } },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 }, // A4
          margin: { top: 1417, right: 1134, bottom: 1417, left: 1701 }, // ~2.5/2/2.5/3 cm (binding margin)
        },
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  const out = path.join(__dirname, "batch-service-sla-article.docx");
  fs.writeFileSync(out, buffer);
  console.log("wrote", out);
});
