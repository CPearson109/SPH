using UnityEngine;

public class FreeCameraController : MonoBehaviour
{
    // Public variables for easy adjustment in the Inspector
    [Header("Movement Settings")]
    public float baseMoveSpeed = 10.0f;    // Normal movement speed
    public float fastMoveSpeed = 80.0f;    // Movement speed when Shift is held down
    public float lookSpeed = 2.0f;         // Mouse look speed
    public float climbSpeed = 10.0f;       // Speed for vertical movement

    private float moveSpeed;                // Current movement speed
    private float yaw = 0.0f;               // Horizontal rotation
    private float pitch = 0.0f;             // Vertical rotation

    void Start()
    {
        // Initialize movement speed
        moveSpeed = baseMoveSpeed;

        // Lock and hide the cursor for an immersive experience
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
    }

    void Update()
    {
        // Toggle cursor lock state with Escape key
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            ToggleCursorLock();
        }

        HandleMovement();
        HandleRotation();
    }

    void HandleMovement()
    {
        // Check if Shift key is held down for increased speed
        if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
        {
            moveSpeed = fastMoveSpeed;
        }
        else
        {
            moveSpeed = baseMoveSpeed;
        }

        // Gather input for movement
        float moveForward = Input.GetAxis("Vertical");    // W/S or Up/Down arrows
        float moveRight = Input.GetAxis("Horizontal");    // A/D or Left/Right arrows
        float moveUp = 0.0f;

        // Vertical movement controls: Space for up, Control for down
        if (Input.GetKey(KeyCode.Space))
        {
            moveUp += 1.0f;
        }
        if (Input.GetKey(KeyCode.LeftControl))
        {
            moveUp -= 1.0f;
        }

        // Calculate movement vector relative to camera orientation
        Vector3 forwardMovement = transform.forward * moveForward;
        Vector3 rightMovement = transform.right * moveRight;
        Vector3 upMovement = transform.up * moveUp;

        // Combine all movement vectors
        Vector3 movement = (forwardMovement + rightMovement) * moveSpeed * Time.deltaTime;
        movement += upMovement * climbSpeed * Time.deltaTime;

        // Apply movement
        transform.position += movement;
    }

    void HandleRotation()
    {
        // Gather mouse movement input
        float mouseX = Input.GetAxis("Mouse X") * lookSpeed;
        float mouseY = Input.GetAxis("Mouse Y") * lookSpeed;

        // Accumulate rotation angles
        yaw += mouseX;
        pitch -= mouseY;
        pitch = Mathf.Clamp(pitch, -90f, 90f); // Prevent flipping

        // Apply rotation to the camera
        transform.eulerAngles = new Vector3(pitch, yaw, 0.0f);
    }

    void ToggleCursorLock()
    {
        if (Cursor.lockState == CursorLockMode.Locked)
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }
        else
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
    }
}
