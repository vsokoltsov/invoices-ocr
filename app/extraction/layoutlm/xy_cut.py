import numpy as np


def recursive_xy_cut(
    boxes: np.ndarray, indices: np.ndarray, min_gap_x: int = 15, min_gap_y: int = 10
) -> list[int]:
  """
    Рекурсивно сортирует индексы bounding box-ов в правильном порядке чтения.

    :param boxes: массив формы (N, 4) с координатами [x0, y0, x1, y1] (в шкале 0-1000)
    :param indices: исходные индексы боксов (от 0 до N-1)
    :param min_gap_x: минимальный горизонтальный зазор между колонками для разреза
    :param min_gap_y: минимальный вертикальный зазор между строками/блоками для разреза
    :return: список отсортированных индексов
    """
  if len(indices) <= 1:
    return indices.tolist()

  current_boxes = boxes[indices]

  # 1. Пробуем разрезать по горизонтали (проекция на ось Y — разделяем строки или крупные секции)
  y_min, y_max = current_boxes[:, 1].min(), current_boxes[:, 3].max()
  y_hist = np.zeros(y_max - y_min + 1, dtype=int)
  for box in current_boxes:
    y_hist[box[1] - y_min : box[3] - y_min] += 1

  # Ищем пустые промежутки (где проекция == 0)
  y_gaps = _find_gaps(y_hist, min_gap_y)
  if y_gaps:
    # Берем самый широкий зазор и делим боксы на "выше" и "ниже"
    cut_y = y_min + y_gaps[0]
    top_mask = current_boxes[:, 3] <= cut_y
    if np.any(top_mask) and np.any(~top_mask):
      return recursive_xy_cut(
          boxes, indices[top_mask], min_gap_x, min_gap_y
      ) + recursive_xy_cut(boxes, indices[~top_mask], min_gap_x, min_gap_y)

  # 2. Если по Y разрезать не удалось, режем по вертикали (проекция на ось X — разделяем колонки)
  x_min, x_max = current_boxes[:, 0].min(), current_boxes[:, 2].max()
  x_hist = np.zeros(x_max - x_min + 1, dtype=int)
  for box in current_boxes:
    x_hist[box[0] - x_min : box[2] - x_min] += 1

  x_gaps = _find_gaps(x_hist, min_gap_x)
  if x_gaps:
    # Берем самый широкий вертикальный зазор и делим на "левую" и "правую" колонки
    cut_x = x_min + x_gaps[0]
    left_mask = current_boxes[:, 2] <= cut_x
    if np.any(left_mask) and np.any(~left_mask):
      return recursive_xy_cut(
          boxes, indices[left_mask], min_gap_x, min_gap_y
      ) + recursive_xy_cut(boxes, indices[~left_mask], min_gap_x, min_gap_y)

  # 3. Базовый случай: дальше делить некуда (например, слова внутри одной строки)
  # Сортируем просто сверху вниз, слева направо
  order = np.lexsort((current_boxes[:, 0], current_boxes[:, 1]))
  return indices[order].tolist()


def _find_gaps(hist: np.ndarray, min_gap: int) -> list[int]:
  """Находит середины пустых участков в гистограмме, отсортированные по ширине зазора."""
  is_zero = hist == 0
  # Находим границы нулевых отрезков
  padded = np.concatenate(([False], is_zero, [False]))
  starts = np.where(padded[1:] & ~padded[:-1])[0]
  ends = np.where(~padded[1:] & padded[:-1])[0]

  gaps = []
  for s, e in zip(starts, ends):
    # Игнорируем пустоту по самым краям
    if s == 0 or e == len(hist):
      continue
    width = e - s
    if width >= min_gap:
      gaps.append((width, (s + e) // 2))

  # Сортируем так, чтобы первыми шли самые широкие разрезы
  gaps.sort(key=lambda x: x[0], reverse=True)
  return [mid for _, mid in gaps]
