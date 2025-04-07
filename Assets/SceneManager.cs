using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneSwitcher : MonoBehaviour
{
    void Update()
    {
        // Check if the "1" key is pressed
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            SceneManager.LoadScene("Scene1"); // Replace with your actual scene name
        }
        // Check if the "2" key is pressed
        else if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            SceneManager.LoadScene("Scene2"); // Replace with your actual scene name
        }
        // Check if the "3" key is pressed
        else if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            SceneManager.LoadScene("Scene3"); // Replace with your actual scene name
        }
        // Check if the "4" key is pressed
        else if (Input.GetKeyDown(KeyCode.Alpha4))
        {
            SceneManager.LoadScene("Scene4"); // Replace with your actual scene name
        }
        // Check if the "5" key is pressed
        else if (Input.GetKeyDown(KeyCode.Alpha5))
        {
            SceneManager.LoadScene("Scene5"); // Replace with your actual scene name
        }
    }
}
