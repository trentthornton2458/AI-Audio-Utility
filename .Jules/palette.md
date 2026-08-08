## 2025-03-06 - Accessible Custom Painted Widgets in PySide6/Qt
**Learning:** Custom drawn visual widgets (inheriting from `QWidget` and overriding `paintEvent`) are completely invisible to screen readers and keyboard navigation by default. They require manual activation of the focus policy (`Qt.FocusPolicy.StrongFocus`), customized drawing of focus indicators, and explicit assignment of screen-reader properties (`setAccessibleName`/`setAccessibleDescription`) to ensure proper accessibility.
**Action:** For any custom-painted Qt widget with interactive features (scrubbing, seeking, dragging, etc.), always apply a strong focus policy, visual focus outlines in the painter, key event overrides (such as Arrow Keys for navigation), and full accessibility descriptions.

## 2025-03-07 - Synchronizing Accessible Properties on State Transitions
**Learning:** In PySide6 / Qt applications, dynamically updating accessible properties (`setAccessibleName`, `setAccessibleDescription`) and tooltips (`setToolTip`) of interactive elements when state transitions occur (e.g., toggling between blind and revealed comparison modes) ensures screen-reader and user accessibility are kept perfectly in sync.
**Action:** Always verify that dynamic labels or text toggles also update their corresponding tooltips and screen-reader helper properties, preserving accessibility in all visual states.
